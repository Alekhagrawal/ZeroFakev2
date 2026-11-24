<!--
File: AGENTS.md
Description: Master AI coding rulebook and architectural guardrails for the ZeroFake project.
Inputs: Referenced by AI coding assistants before creating or modifying any project file.
Outputs: Enforces 7 strict scientific, engineering, and documentation standards across the repository.
-->

# ZeroFake Project — AI Agent Guardrails & Strict Rules

Every file created or modified in this repository **must** strictly adhere to the following 7 project rules derived from `ZeroFake_Zero_Day_Deepfake_Detection_Project.pdf`:

---

## 1. Focused Scope (Sec 18)
* **Build only the Phase-1 scope:**
  * **Backbone:** Pretrained `ConvNeXt-Tiny` (`timm`) outputting 768-d features, followed by a 2-layer MLP projection head mapping to L2-normalized 128-d embeddings $z$.
  * **Multi-Generator Supervised Contrastive Learning:** Supervised Contrastive Loss ($\mathcal{L}_{\text{con}}$, temperature $\tau = 0.07$) with generator-balanced mini-batch sampling (50% Real crops, 50% Fake crops evenly distributed across training generators).
  * **Multi-Prototype Fake Representation:** Learnable prototype bank consisting of 1 Real prototype and $K = 6$ Fake prototypes using cosine distance ($\mathcal{L}_{\text{proto}}$), combined with binary classification loss as $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.5\mathcal{L}_{\text{con}} + 0.2\mathcal{L}_{\text{proto}}$.
  * **Calibrated Open-Set Uncertainty:** Combined uncertainty score $U(x) = \alpha D_{\text{prototype}}(x) + \beta H(p(y|x)) + \gamma S_{\text{OOD}}(x)$ supporting a 3-way decision rule (`REAL`, `FAKE`, or `UNKNOWN`).
* **Strict Exclusion:** Skip the optional adversarial gradient-reversal extension. Do not add unrequested architectures or complexity outside Phase-1.

## 2. Zero Data Leakage (Sec 5.1)
* **Strict Identity & Video Partitioning:** Never allow frames from the same source video or person identity in both `train` and `val`/`test` splits.
* **Swapped Pair Isolation:** When parsing manipulated video filenames (e.g., `000_003.mp4` in FaceForensics++), extract **both** source and target person identity IDs (`000` and `003`) and group connected components so no individual's face ever crosses partition boundaries (`train`, `val`, `test_closed`, `test_unseen`).
* **Automated Verification:** Every splitting routine must run an automatic programmatic assertion verifying zero identity or video overlap across splits before saving `splits.json`.

## 3. Unseen-Generator Isolation (Sec 19)
* **Strict Leave-One-Generator-Out (LOGO) Holdout:** Never use the held-out unseen generator (`test_unseen`, e.g., `NeuralTextures`) or cross-dataset benchmark (`Celeb-DF v2`) for model selection, hyperparameter tuning, z-score normalization, or threshold calibration.
* **Validation-Only Calibration:** Fit all feature cluster statistics, z-score normalization parameters for $U(x)$ terms, and the open-set rejection threshold $\tau_{\text{reject}}$ (targeting 95% validation coverage) **strictly on the `val` split**. Evaluate `test_unseen` only during final evaluation.

## 4. Cloud GPU Ready
* **GitHub Repository URL:** Always use the official repository URL `https://github.com/Alekhagrawal/ZeroFake.git` in every Colab notebook (`01_Prepare_Data.ipynb`, `02_Train_ZeroFake.ipynb`, etc.) when cloning or pulling code into `/content/ZeroFake`. Use native Colab commands (`!git clone`, `%cd /content/ZeroFake`, `!pip install --no-deps facenet-pytorch`, and `!python -u ...`) so logs stream live in Google Colab without buffering.
* **Mixed Precision:** Use PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast` / `torch.amp.autocast` with `fp16` and `GradScaler`) during training and inference on CUDA GPUs, with graceful fallback to `fp32` on CPU.
* **Fast Local SSD I/O:** Store cropped faces in a single `cropped_faces.zip` archive on Google Drive (`ZeroFake_Storage/`) and unzip to local `/content/data` SSD at Colab startup to avoid Google Drive network latency.
* **Per-Epoch Checkpointing & Resume:** Save `last_checkpoint.pt` and `best_model.pt` (tracked via validation ROC-AUC) after **every epoch** (including model weights, optimizer state, scheduler state, scaler state, and epoch index) so interrupted Colab runs can resume seamlessly via `--resume`.

## 5. Mandatory Dry-Run
* **`--dry-run` Flag in Every Python Script:** Every executable Python script (`preprocessing/`, `models/`, `losses/`, `train.py`, `evaluate.py`, `app.py`, etc.) must support a `--dry-run` CLI flag via `argparse`.
* **Under 5 Seconds on CPU:** When `--dry-run` is passed, the script must synthesize tiny dummy tensors or mock directory structures in memory/temp storage, verify full forward/backward passes, file I/O, or pipeline assertions on CPU, and exit cleanly with zero errors in **under 5 seconds** before Colab upload.

## 6. Beginner Comments
* **Top-of-File Header:** Add plain-English comments at the very top of every file explaining:
  1. **What the file does** in simple, non-technical terms.
  2. **Inputs** (expected files, arguments, or tensor shapes).
  3. **Outputs** (returned tensors, saved files, or console outputs).
* **Inline Explanations:** Include clear, beginner-friendly comments above key classes, functions, and mathematical formulas so a non-coder can easily understand and explain the code during a viva presentation.

## 7. Summary
* **Post-Task File Summary:** After completing every task, always provide a clear, concise summary of all created or changed files, explaining what was added or modified and how it satisfies these rules.
