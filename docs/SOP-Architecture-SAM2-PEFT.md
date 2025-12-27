# SAM 2 + LoRA Implementation SOP
## 1. Why PEFT?
- Full fine-tuning SAM 2 requires >80GB VRAM. A40 (48GB) must use LoRA.
## 2. LoRA Logic
- Freeze pre-trained weights $W_0$.
- Inject rank decomposition matrices $A$ and $B$: $W = W_0 + \Delta W = W_0 + BA$.
- Target layers: Attention blocks in Image Encoder.
## 3. Auto-Prompting Strategy
- SAM 2 needs prompts. Use a coarse segmentor (e.g., standard U-Net or YOLO) to generate Bounding Boxes.
- Pipeline: Input -> Coarse Net -> BBox -> SAM 2 (LoRA) -> Fine Mask.
