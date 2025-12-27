"""
Swin-UMamba Trainer with Boundary DoU Loss.

This trainer extends nnUNetTrainerSwinUMamba with:
1. Boundary DoU Loss for improved boundary delineation
2. Mixed precision training (torch.cuda.amp) - enabled by default
3. Configurable loss weights

Reference: docs/SOP-Architecture-SwinUMamba.md
Hardware: Optimized for NVIDIA A40 (48GB)
"""
import numpy as np
import torch

from nnunetv2.training.loss.boundary_loss import DC_and_Boundary_loss, BoundaryDoULoss
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUMamba import nnUNetTrainerSwinUMamba


class nnUNetTrainerSwinUMambaBoundary(nnUNetTrainerSwinUMamba):
    """
    Swin-UMamba Trainer with Boundary DoU Loss.

    Key features:
    - Inherits all Swin-UMamba functionality (pretrained weights, freeze encoder, etc.)
    - Adds Boundary DoU Loss for better boundary segmentation
    - Configurable loss weights via class attributes

    Loss function:
        L = w_ce * CE + w_dice * Dice + w_boundary * BoundaryDoU

    Default weights:
        - weight_ce = 1.0
        - weight_dice = 1.0
        - weight_boundary = 0.5

    Usage:
        nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerSwinUMambaBoundary
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda')
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Loss weights - can be overridden in subclasses
        self.weight_ce = 1.0
        self.weight_dice = 1.0
        self.weight_boundary = 0.5

        # Boundary extraction kernel size (for morphological dilation)
        self.boundary_kernel_size = 3

    def _build_loss(self):
        """
        Build the combined Dice + CE + Boundary loss function.

        For region-based labels (has_regions=True), uses DC_and_BCE_loss
        with additional boundary loss wrapper.

        For standard labels, uses DC_and_Boundary_loss which combines
        Dice, CrossEntropy, and BoundaryDoU losses.
        """
        if self.label_manager.has_regions:
            # Region-based: wrap BCE+Dice with boundary loss
            base_loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
            loss = _BoundaryLossWrapper(
                base_loss=base_loss,
                weight_base=self.weight_ce + self.weight_dice,
                weight_boundary=self.weight_boundary,
                boundary_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp,
                    'kernel_size': self.boundary_kernel_size
                }
            )
        else:
            # Standard labels: use combined loss
            loss = DC_and_Boundary_loss(
                soft_dice_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': False,
                    'ddp': self.is_ddp
                },
                ce_kwargs={},
                boundary_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': False,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp,
                    'kernel_size': self.boundary_kernel_size
                },
                weight_ce=self.weight_ce,
                weight_dice=self.weight_dice,
                weight_boundary=self.weight_boundary,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        # Deep supervision wrapper
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # Exponentially decreasing weights for each resolution
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0  # Don't use lowest resolution

            # Normalize weights
            weights = weights / weights.sum()

            loss = DeepSupervisionWrapper(loss, weights)

        self.print_to_log_file(
            f"Using Boundary Loss with weights: "
            f"CE={self.weight_ce}, Dice={self.weight_dice}, Boundary={self.weight_boundary}"
        )

        return loss


class _BoundaryLossWrapper(torch.nn.Module):
    """
    Wrapper to add boundary loss to an existing base loss.

    Used for region-based labels where we need to combine
    DC_and_BCE_loss with BoundaryDoULoss.
    """

    def __init__(
        self,
        base_loss: torch.nn.Module,
        weight_base: float,
        weight_boundary: float,
        boundary_kwargs: dict
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weight_base = weight_base
        self.weight_boundary = weight_boundary

        from nnunetv2.utilities.helpers import softmax_helper_dim1
        self.boundary_loss = BoundaryDoULoss(
            apply_nonlin=torch.sigmoid,  # BCE uses sigmoid
            **boundary_kwargs
        )

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(net_output, target)
        boundary = self.boundary_loss(net_output, target)
        return self.weight_base * base + self.weight_boundary * boundary


# ============================================================================
# Variant Trainers with different loss weight configurations
# ============================================================================

class nnUNetTrainerSwinUMambaBoundaryHeavy(nnUNetTrainerSwinUMambaBoundary):
    """
    Variant with higher boundary loss weight (1.0).

    Useful when boundary accuracy is critical, e.g., for small structures
    or when HD95 metric is prioritized.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_boundary = 1.0


class nnUNetTrainerSwinUMambaBoundaryLight(nnUNetTrainerSwinUMambaBoundary):
    """
    Variant with lower boundary loss weight (0.25).

    Useful when overall Dice is more important than boundary precision.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_boundary = 0.25
