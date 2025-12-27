# Swin-UMamba Implementation SOP
## 1. Baseline
- Load ImageNet weights (`vmamba_tiny_e292.pth`).
- Use `use_pretrain=True`.
## 2. Innovation: Boundary DoU Loss
- Math: $Loss = 1 - \frac{Inter_{boundary}}{Union_{boundary}}$
- Use `max_pool3d` for dilation to extract boundaries.
## 3. Innovation: MC Dropout
- Insert `nn.Dropout3d(p=0.1)` in Decoder upsampling layers.
- Inference: Run T=20 forward passes to get uncertainty map.
## 4. Advanced: Semi-Supervised Learning
- Use Uncertainty Map from MC Dropout.
- Logic: If uncertainty < threshold, use prediction as Pseudo-label for unlabeled data.
- Benefit: Utilize large unlabeled portion of BraTS dataset.
