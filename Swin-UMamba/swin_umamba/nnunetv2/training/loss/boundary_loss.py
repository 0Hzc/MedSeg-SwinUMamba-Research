"""
Boundary DoU (Dice over Union) Loss for medical image segmentation.

Reference: SOP-Architecture-SwinUMamba.md
Math: Loss = 1 - Inter_boundary / Union_boundary
Uses max_pool3d for morphological dilation to extract boundaries.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class BoundaryDoULoss(nn.Module):
    """
    Boundary Dice-over-Union Loss.

    Computes the DoU loss specifically on boundary regions, which helps
    the model focus on accurate boundary delineation.

    Args:
        apply_nonlin: Nonlinearity to apply to predictions (e.g., softmax)
        batch_dice: Whether to compute dice over entire batch
        do_bg: Whether to include background class
        smooth: Smoothing factor to avoid division by zero
        ddp: Whether using distributed data parallel
        kernel_size: Size of the dilation kernel for boundary extraction
    """

    def __init__(
        self,
        apply_nonlin: Optional[Callable] = None,
        batch_dice: bool = False,
        do_bg: bool = False,
        smooth: float = 1e-5,
        ddp: bool = True,
        kernel_size: int = 3
    ):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.kernel_size = kernel_size

    def _extract_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary regions using morphological dilation.

        Uses max_pool3d (or max_pool2d for 2D) to dilate the mask,
        then subtracts the original to get the boundary.

        Args:
            x: Input tensor of shape (B, C, D, H, W) or (B, C, H, W)

        Returns:
            Boundary tensor of same shape as input
        """
        ndim = x.ndim - 2  # Spatial dimensions (2 or 3)
        padding = self.kernel_size // 2

        if ndim == 3:
            # 3D case: use max_pool3d
            dilated = F.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding
            )
        elif ndim == 2:
            # 2D case: use max_pool2d
            dilated = F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding
            )
        else:
            raise ValueError(f"Unsupported spatial dimensions: {ndim}")

        # Boundary = dilated - original (only positive part)
        boundary = torch.clamp(dilated - x, min=0, max=1)
        return boundary

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Boundary DoU Loss.

        Args:
            net_output: Network predictions (B, C, ...)
            target: Ground truth labels (B, 1, ...) or (B, ...)
            loss_mask: Optional mask for valid pixels

        Returns:
            Scalar loss value
        """
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # Spatial axes for reduction
        axes = tuple(range(2, net_output.ndim))

        # Convert target to one-hot encoding if necessary
        with torch.no_grad():
            if net_output.ndim != target.ndim:
                target = target.view((target.shape[0], 1, *target.shape[1:]))

            if net_output.shape == target.shape:
                y_onehot = target
            else:
                y_onehot = torch.zeros(
                    net_output.shape,
                    device=net_output.device,
                    dtype=torch.float32
                )
                y_onehot.scatter_(1, target.long(), 1)

            # Extract boundaries from ground truth
            gt_boundary = self._extract_boundary(y_onehot)

            if not self.do_bg:
                gt_boundary = gt_boundary[:, 1:]
                y_onehot = y_onehot[:, 1:]

        # Extract boundaries from predictions
        if not self.do_bg:
            pred = net_output[:, 1:]
        else:
            pred = net_output

        pred_boundary = self._extract_boundary(pred)

        # Apply loss mask if provided
        if loss_mask is not None:
            pred_boundary = pred_boundary * loss_mask
            gt_boundary = gt_boundary * loss_mask

        # Compute intersection and union on boundary regions
        intersection = (pred_boundary * gt_boundary).sum(axes)
        union = pred_boundary.sum(axes) + gt_boundary.sum(axes) - intersection

        # Handle batch dice
        if self.batch_dice:
            if self.ddp:
                intersection = AllGatherGrad.apply(intersection).sum(0)
                union = AllGatherGrad.apply(union).sum(0)
            intersection = intersection.sum(0)
            union = union.sum(0)

        # Compute DoU: Inter / Union
        dou = (intersection + self.smooth) / (union + self.smooth)

        # Loss = 1 - DoU
        loss = 1.0 - dou.mean()

        return loss


class DC_and_Boundary_loss(nn.Module):
    """
    Combined Dice + CrossEntropy + Boundary DoU Loss.

    This compound loss combines the standard DC+CE loss with the
    boundary-focused DoU loss for improved boundary delineation.

    Args:
        soft_dice_kwargs: Kwargs for SoftDiceLoss
        ce_kwargs: Kwargs for CrossEntropyLoss
        boundary_kwargs: Kwargs for BoundaryDoULoss
        weight_ce: Weight for CE loss
        weight_dice: Weight for Dice loss
        weight_boundary: Weight for Boundary loss
        ignore_label: Label to ignore in loss computation
    """

    def __init__(
        self,
        soft_dice_kwargs: dict,
        ce_kwargs: dict,
        boundary_kwargs: Optional[dict] = None,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        weight_boundary: float = 0.5,
        ignore_label: Optional[int] = None,
        dice_class=None
    ):
        super().__init__()

        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
        from nnunetv2.utilities.helpers import softmax_helper_dim1

        if dice_class is None:
            dice_class = MemoryEfficientSoftDiceLoss

        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        self.ignore_label = ignore_label

        # Standard losses
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

        # Boundary loss
        if boundary_kwargs is None:
            boundary_kwargs = {}
        self.boundary = BoundaryDoULoss(
            apply_nonlin=softmax_helper_dim1,
            **boundary_kwargs
        )

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            net_output: Network predictions (B, C, ...)
            target: Ground truth (B, 1, ...)

        Returns:
            Combined scalar loss
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # Dice loss
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0

        # CE loss
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        # Boundary loss
        boundary_loss = self.boundary(net_output, target_dice, loss_mask=mask) \
            if self.weight_boundary != 0 else 0

        result = (
            self.weight_ce * ce_loss +
            self.weight_dice * dc_loss +
            self.weight_boundary * boundary_loss
        )

        return result


if __name__ == '__main__':
    # Quick test
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    # 3D test
    pred = torch.rand((2, 4, 16, 32, 32))  # B, C, D, H, W
    target = torch.randint(0, 4, (2, 16, 32, 32))  # B, D, H, W

    loss_fn = BoundaryDoULoss(
        apply_nonlin=softmax_helper_dim1,
        batch_dice=False,
        do_bg=False,
        ddp=False
    )

    loss = loss_fn(pred, target)
    print(f"Boundary DoU Loss (3D): {loss.item():.4f}")

    # 2D test
    pred_2d = torch.rand((2, 4, 128, 128))
    target_2d = torch.randint(0, 4, (2, 128, 128))

    loss_2d = loss_fn(pred_2d, target_2d)
    print(f"Boundary DoU Loss (2D): {loss_2d.item():.4f}")
