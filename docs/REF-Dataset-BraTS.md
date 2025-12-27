# BraTS 2024 Dataset Specs
## 1. Modalities
- Channel 0: T1
- Channel 1: T1ce (Contrast-enhanced)
- Channel 2: T2
- Channel 3: FLAIR
## 2. Labels
- Label 1: NCR/NET (Necrotic/Non-enhancing tumor core)
- Label 2: ED (Peritumoral Edema)
- Label 3: ET (Enhancing Tumor)
## 3. Challenge: Domain Adaptation
- BraTS-Africa data has lower quality/different field strength.
- Experiment: Train on Standard BraTS -> Test on Africa -> Apply LoRA to adapt.
