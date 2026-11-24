<!--
File: SLIDES_OUTLINE.md
Description: 12-slide PowerPoint presentation blueprint for the ZeroFake project defense.
Inputs: Actual experimental metrics from results/ablation_summary.csv and results/robustness_summary.csv plus the 4 300-DPI PNG charts in results/plots/.
Outputs: Exact slide titles, 3-4 concise presentation bullet points per slide, and explicit instructions on which PNG chart or table to place on each slide.
-->

# 📊 ZeroFake — 12-Slide PowerPoint Presentation Blueprint

Use this slide-by-slide blueprint to build your final 12-slide presentation deck. Every metric cited below comes directly from [`results/ablation_summary.csv`](results/ablation_summary.csv) and [`results/robustness_summary.csv`](results/robustness_summary.csv), and all four high-resolution 300-DPI PNG plots are located in [`results/plots/`](results/plots/).

---

## Slide 1: Title & Project Overview
* **Slide Title:** **ZeroFake: Zero-Day Deepfake Detection via Multi-Prototype Contrastive Learning & Calibrated Open-Set Uncertainty**
* **Bullet Points:**
  * **Objective:** Detect deepfakes from known manipulation tools while safely flagging unseen ("zero-day") generators and corrupted media.
  * **Core Innovation:** Upgrades traditional 2-way binary detection (`REAL` vs. `FAKE`) to a **Calibrated 3-Way Open-Set Decision Rule (`REAL`, `FAKE`, or `UNKNOWN`)**.
  * **Architecture Stack:** Pretrained `ConvNeXt-Tiny` (`768-d`) + `128-d` Multi-Generator Supervised Contrastive Space + `1 Real & 6 Fake` Learnable Prototypes.
  * **Headline Result:** Achieves **0.9933 Closed-Set ROC-AUC**, **98.64% Selective Accuracy** (at **94.48% coverage**), and boosts Out-of-Distribution detection **AUROC from 0.2960 to 0.6252** (**0.8016 OOD AUPR**).
* **Visual to Place on Slide:** Screenshot of the live **Gradio Web App (`app.py`)** showing the 3-way verdict badge and architecture header.

---

## Slide 2: Problem Statement — Why Standard Deepfake Detectors Fail
* **Slide Title:** **The Zero-Day Deepfake Challenge: Why Binary Classifiers Are Not Enough**
* **Bullet Points:**
  * **Closed-Set Assumption Failure:** Standard detectors assume test-time deepfakes come from the exact same manipulation tools seen during training.
  * **Single-Cluster Overfitting:** Forcing diverse synthesis families (autoencoder swaps, graphics meshes, neural rendering) into a single "Fake" class causes models to memorize generator-specific shortcuts.
  * **Overconfident Blind Guesses:** When faced with an unseen zero-day generator (like *NeuralTextures*) or heavy sensor noise, a binary classifier cannot say *"I don't know"*—producing silent false negatives.
  * **Our Goal:** Build a detector that clusters authentic faces tightly, models multiple distinct fake families, and abstains (`UNKNOWN`) when predictive uncertainty is high.
* **Visual to Place on Slide:** Simple conceptual comparison diagram: **Standard 2-Way Classifier (`REAL` / `FAKE`)** vs. **ZeroFake 3-Way Open-Set Detector (`REAL` / `FAKE` / `UNKNOWN`)**.

---

## Slide 3: Dataset & Zero-Leakage Leave-One-Generator-Out (LOGO) Protocol
* **Slide Title:** **Leakage-Free Dataset Protocol: Connected-Component Identity Splitting & LOGO**
* **Bullet Points:**
  * **Benchmark Dataset:** FaceForensics++ (`c23` compressed real-world videos) containing `Real` videos plus 4 manipulation methods: *Deepfakes*, *Face2Face*, *FaceSwap*, and *NeuralTextures*.
  * **Swapped-Pair Identity Isolation (`create_splits.py`):** Fake filenames like `000_003.mp4` involve **two** actors (`000` and `003`). We group all linked actors via a **Disjoint-Set Connected-Components graph** so no individual's face ever crosses between `train` (80%), `val` (10%), and `test_closed` (10%).
  * **Strict Unseen-Generator Holdout:** **100% of `NeuralTextures`** is locked inside `test_unseen` (zero frames in `train` or `val`) to simulate a zero-day deepfake tool.
  * **MTCNN $1.3\times$-Margin Preprocessing (`extract_and_crop.py`):** Extracts 16 evenly spaced frames/video, aligns eyes horizontally, expands the face box by **$1.3\times$** to capture jawline/cheek blending seams, and resizes to $224 \times 224$ RGB.
* **Visual to Place on Slide:** Table or diagram illustrating the **Connected-Component Actor Split (`80% train / 10% val / 10% test_closed`) + 100% Held-Out `test_unseen` (`NeuralTextures`)**.

---

## Slide 4: End-to-End ZeroFake Architecture
* **Slide Title:** **ZeroFake System Architecture: From $224 \times 224$ Face Crop to 3-Way Verdict**
* **Bullet Points:**
  * **Backbone (`models/backbone.py`):** Pretrained `ConvNeXt-Tiny` (`timm`) mapping $224 \times 224 \times 3$ face crops to rich $768\text{-d}$ spatial feature vectors $h$.
  * **Contrastive Projection Head (`models/projection_head.py`):** 2-layer MLP (`768 -> 256 -> 128`) with ReLU and $L_2$-normalization projecting features onto the $128\text{-d}$ unit hypersphere ($\|z\|_2 = 1$).
  * **Dual Decision Branches:**
    1. **Binary Classifier Head (`models/classifier.py`):** Predicts fake probability $p = \sigma(s) \in [0, 1]$.
    2. **Multi-Prototype Layer (`models/prototype.py`):** Computes cosine distances $d_{\text{real}}(z)$ to **1 Real Prototype** and $d_{\text{fake}}(z) = \min_{k=1..6} d(z, p_{\text{fake}, k})$ across **$K = 6$ Fake Prototypes**.
  * **Generator-Balanced Sampling (`datasets/deepfake_dataset.py`):** Every batch of 64 images contains exactly 50% Real crops and 50% Fake crops split evenly across all seen training generators.
* **Visual to Place on Slide:** System pipeline block diagram from [`README.md`](README.md) / [`SPEC.md`](SPEC.md).

---

## Slide 5: Multi-Generator Contrastive & Multi-Prototype Losses
* **Slide Title:** **Mathematical Formulation: Joint Contrastive & Multi-Prototype Learning**
* **Bullet Points:**
  * **Combined Training Objective (`losses/total_loss.py`):**
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + 0.5\,\mathcal{L}_{\text{con}} + 0.2\,\mathcal{L}_{\text{proto}}$$
  * **Supervised Contrastive Loss ($\mathcal{L}_{\text{con}}, \tau = 0.07$):** Pulls all authentic `Real` faces into one tight cluster and aligns `Fake` representations across generators on the unit hypersphere.
  * **Multi-Prototype Pull, Margin & Diversity Loss ($\mathcal{L}_{\text{proto}}$):** Pulls each embedding toward its closest same-class prototype, enforces a $0.5$ cosine margin against opposite-class prototypes, and penalizes pairwise similarity among the $K = 6$ fake prototypes so they never collapse.
  * **Mixed-Precision Engineering:** Runs the ConvNeXt-Tiny forward pass in fast `fp16` CUDA `autocast` and casts outputs to `float32` outside `autocast` for overflow-free loss and uncertainty calculation.
* **Visual to Place on Slide:** **`results/plots/tsne_prototypes.png`** ([`results/plots/tsne_prototypes.png`](file:///e:/Deepfake%20ai/ZeroFake/results/plots/tsne_prototypes.png)) — *Shows the 2D t-SNE projection of 128-d test embeddings alongside the learned 1 Real Prototype star and 6 Fake Prototype stars.*

---

## Slide 6: Calibrated Open-Set Uncertainty & 3-Way Decision Rule
* **Slide Title:** **Calibrated Open-Set Uncertainty $U(x)$ & Strict Validation-Only Thresholding**
* **Bullet Points:**
  * **Three Complementary Uncertainty Signals (`models/open_set.py`):**
    1. **Nearest-Prototype Cosine Distance ($\alpha = 0.4$):** $D_{\text{prototype}}(x) = \min(d_{\text{real}}(z), d_{\text{fake}}(z))$.
    2. **Predictive Shannon Entropy ($\beta = 0.3$):** $H(p(y|x))$ measuring classifier boundary ambiguity.
    3. **Out-of-Distribution Centroid Distance ($\gamma = 0.3$):** $S_{\text{OOD}}(x)$ measuring distance to training class feature centroids.
  * **Validation-Only Z-Score Calibration (Rule 3):** Means, standard deviations, and the rejection cutoff $\tau_{\text{reject}} = 0.2756$ are calibrated **strictly at the 95th percentile of the `val` split**—never touching `test_unseen`.
  * **3-Way Selective Decision Rule:**
    * Output **`UNKNOWN`** if $U(x) > \tau_{\text{reject}}$; otherwise output **`FAKE`** ($E_{\text{fake}}(x) \ge 0.5$) or **`REAL`** ($E_{\text{fake}}(x) < 0.5$).
* **Visual to Place on Slide:** **`results/plots/confusion_matrices.png`** ([`results/plots/confusion_matrices.png`](file:///e:/Deepfake%20ai/ZeroFake/results/plots/confusion_matrices.png)) — *Side-by-side comparison of the Baseline `2x2` confusion matrix vs. Full ZeroFake `2x3` (`REAL`, `FAKE`, `UNKNOWN`) confusion matrix.*

---

## Slide 7: Ablation Study Results (Configs A through E)
* **Slide Title:** **Ablation Study: Quantifying the Gain of Each ZeroFake Component**
* **Bullet Points:**
  * **High Closed-Set Precision Across All Models:** On known manipulation generators (`test_closed`), ZeroFake achieves **0.9933 ROC-AUC**, **0.9835 F1-score**, and **0.0250 Equal Error Rate (EER)**.
  * **+2.03% Selective Accuracy Improvement:** By filtering out the ~5.5% highest-uncertainty predictions ($\tau_{\text{reject}} = 0.2756$, **94.48% coverage**), **Config E (Full ZeroFake)** reaches **98.64% Selective Accuracy** (vs. **96.61%** on Config A Baseline).
  * **More Than Doubles OOD Detection AUROC:** Baseline (`0.2960`) and Contrastive-only (`0.2036`) struggle to rank unseen/OOD uncertainty; adding prototype distances in Configs C and D/E jumps **OOD AUROC to `0.6252–0.6413`** and **OOD AUPR to `0.8016–0.8158`**.
* **Visual to Place on Slide:** Formatted **Ablation Comparison Table (Configs A–E)** from [`results/ablation_summary.csv`](file:///e:/Deepfake%20ai/ZeroFake/results/ablation_summary.csv).

---

## Slide 8: Unseen-Generator ROC Comparison (`NeuralTextures` Holdout)
* **Slide Title:** **Zero-Day Generalization Analysis on Held-Out `NeuralTextures`**
* **Bullet Points:**
  * **Why `NeuralTextures` Is Challenging:** Unlike full-face swaps (*Deepfakes*, *FaceSwap*), *NeuralTextures* applies a subtle neural rendering patch primarily around the mouth, creating a severe domain shift (`unseen_roc_auc = 0.5691–0.6646`).
  * **Failure of Fixed 0.5 Binary Thresholds:** Standard closed-set detectors tuned on high-artifact face swaps assign low fake probabilities to subtle neural-rendered fakes, resulting in low fixed-threshold recall (`2.54%–6.17%`).
  * **The Case for Open-Set Uncertainty:** Because raw binary scores shift under unseen generators, combining prototype distance $D_{\text{prototype}}(x)$ and $S_{\text{OOD}}(x)$ provides a reliable **0.8016 OOD AUPR** signal to flag unfamiliar manipulations.
* **Visual to Place on Slide:** **`results/plots/roc_comparison.png`** ([`results/plots/roc_comparison.png`](file:///e:/Deepfake%20ai/ZeroFake/results/plots/roc_comparison.png)) — *Overlaid 300-DPI ROC curves comparing Baseline, Contrastive, Single-Prototype, and Full ZeroFake on `test_unseen`.*

---

## Slide 9: Robustness Stress Tests Under Real-World Degradations
* **Slide Title:** **In-Memory Robustness Stress Tests: Compression, Blur, Resize & Illumination**
* **Bullet Points:**
  * **12 Real-World Distortions Tested in RAM (`evaluation/robustness.py`):** Evaluated Clean vs. JPEG (`Q90`, `Q75`, `Q50`), Gaussian Blur ($3 \times 3$, $5 \times 5$), Gaussian Noise (`0.03`, `0.06`), Low-Res Downsampling ($112 \times 112 \rightarrow 224 \times 224$), and Brightness/Contrast shifts ($\pm 20\%$).
  * **Strong Compression & Downsampling Resilience:** On `test_closed`, ZeroFake maintains **0.9895 ROC-AUC (96.77% Selective Acc)** at JPEG Q75, **0.9780 ROC-AUC (91.29% Selective Acc)** at heavy JPEG Q50, and **0.9856 ROC-AUC (97.29% Selective Acc)** after $112 \times 112$ downsampling.
  * **Lighting & Blur Stability:** Maintains **0.9889 ROC-AUC** under $3 \times 3$ Gaussian blur and **0.9841–0.9932 ROC-AUC** across all $\pm 20\%$ brightness and contrast perturbations.
  * **Automatic Abstention on Sensor Noise:** Under high-frequency Gaussian noise (`std=0.03`), $U(x)$ automatically increases `UNKNOWN` rejections—dropping coverage from **94.48% to 71.75%** (and **62.36%** on `test_unseen`) to protect against corrupted inputs.
* **Visual to Place on Slide:** Summary table or bar chart of the **Robustness Stress-Test Results** from [`results/robustness_summary.csv`](file:///e:/Deepfake%20ai/ZeroFake/results/robustness_summary.csv).

---

## Slide 10: Explainable AI — ConvNeXt-Tiny Grad-CAM Facial Heatmaps
* **Slide Title:** **Visual Explainability: Where Does ZeroFake Look on the Face?**
* **Bullet Points:**
  * **Spatial Attribution via Grad-CAM (`visualization/gradcam.py`):** Extracts gradient-weighted activation maps from the final convolutional stage of `ConvNeXt-Tiny`.
  * **Authentic (`REAL`) Faces:** Attention distributes smoothly across natural central facial landmarks (eyes, nose bridge, and cheeks).
  * **Manipulated (`FAKE`) Faces:** Heatmaps lock onto high-artifact manipulation regions—specifically the **mouth/lip synthesis zone** and **peri-orbital/jawline blending boundaries**.
  * **Rejected (`UNKNOWN`) Faces:** Diffuse or off-landmark activation patterns trigger elevated prototype and OOD distances ($U(x) > 0.2756$), explaining why the frame is flagged for human review.
* **Visual to Place on Slide:** **`results/plots/gradcam_heatmaps.png`** ([`results/plots/gradcam_heatmaps.png`](file:///e:/Deepfake%20ai/ZeroFake/results/plots/gradcam_heatmaps.png)) — *The 300-DPI $4 \times 3$ grid (`Original Face | Grad-CAM Heatmap | Spatial Attention Overlay`) across Real, Known Fake, Unseen Fake, and `UNKNOWN` faces.*

---

## Slide 11: Live Interactive Gradio Web Application (`app.py`)
* **Slide Title:** **Live Forensic Dashboard: Real-Time Image & Video Deepfake Analysis**
* **Bullet Points:**
  * **Multi-Modal Input (`app.py`):** Accepts any uploaded face photograph or short `.mp4` video clip (sampling 8 evenly spaced frames for ~2-second turnaround), plus **4 one-click sample buttons** (*Real Face*, *Known Fake*, *Unseen Generator Fake*, *Blurry/Ambiguous Face*).
  * **Crash-Proof MTCNN Gate:** Automatically detects and aligns the primary face with a $1.3\times$ margin; displays a friendly `⚠️ NO FACE DETECTED` banner on faceless frames without crashing.
  * **Four Real-Time Visual Panels:**
    1. Color-coded 3-way verdict badge (**`REAL`** green, **`FAKE`** red, **`UNKNOWN`** amber).
    2. Live uncertainty gauge comparing $U(x)$ against $\tau_{\text{reject}} = 0.2756$.
    3. Cosine similarity bar chart across the **1 Real Prototype and all 6 Fake Prototypes**.
    4. Side-by-side Original Crop, Grad-CAM Heatmap, and Artifact Overlay.
* **Visual to Place on Slide:** Live demonstration (or 2-minute backup MP4 recording / full-window screenshot of [`app.py`](file:///e:/Deepfake%20ai/ZeroFake/app.py)).

---

## Slide 12: Conclusion, Engineering Best Practices & Future Scope
* **Slide Title:** **Conclusion & Future Scope**
* **Bullet Points:**
  * **Scientific Summary:** ZeroFake demonstrates that combining **Multi-Generator Contrastive Learning** with **Multi-Prototype Geometry** and **Validation-Calibrated Open-Set Uncertainty** boosts selective accuracy to **98.64%** and more than doubles OOD detection AUROC (**0.2960 $\rightarrow$ 0.6252**).
  * **Reproducible Cloud Engineering:** 100% modular PyTorch codebase with `<5s` CPU `--dry-run` verification in every script, zero-leakage connected-component dataset splitting, and automated Google Colab Pro GPU notebooks.
  * **Future Scope (Phase-2 Extensions):**
    * Incorporate temporal frame-to-frame optical flow consistency across video clips.
    * Expand unseen-generator testing to diffusion-based talking-head generators (e.g., Stable Video Diffusion, Sora-style syntheses) and cross-dataset Celeb-DF v2 zero-shot benchmarks.
* **Visual to Place on Slide:** Summary key-metrics callout box (`98.64% Selective Acc` | `0.9933 Closed AUC` | `0.8016 OOD AUPR` | `100% Zero Identity Leakage`) + GitHub QR code / link (`https://github.com/Alekhagrawal/ZeroFake`).
