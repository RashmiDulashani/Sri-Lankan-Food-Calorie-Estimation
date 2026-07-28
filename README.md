# 🥘 Sri Lankan Food Recognition and Calorie Estimation System

> Computer Vision System for identifying Sri Lankan meals and estimating their
> calorie content, using a two-headed EfficientNet-B0 model with portion-aware
> calorie estimation.

**Module:** IT41043 — Intelligent Systems | Horizon Campus 
**Team:** G.Rashmi Dulashani (ITBIN-2313-0031) & H.M.Imashi Dilshani (ITBIN-2313-0025)
**Module Leader:** Mr. Isuru Madusanka Samarappulige

---

## ❓ Research Question
Does transfer learning on a locally collected Sri Lankan food image dataset give
significantly better food recognition and calorie estimation than models trained
only on international datasets, for complex Sri Lankan mixed meals?

## 🧩 Research Gap
Existing food recognition and calorie estimation systems are
trained and tested on structured, clearly separated Western or East Asian meals.
Sri Lankan meals (rice and curry, kottu, hoppers, string hoppers) are visually
complex, mixed on one plate, and vary widely in portion size — no existing
dataset or model has been tested against this specific challenge.

## 📊 Project Status

| Milestone | Status | Notes |
|---|---|---|
| M1 — Research Proposal | ✅ Complete | Gap, Question, and Scope Approved |
| M2 — Methodology & Data Description | 🔄 In progress | This Repository reflects Current M2 State |
| Data Collection | 🔄 In progress | Target: ~3,000–4,000 Images, 18–20 Dish Classes |
| M3 — Implementation | ⬜ Not Started | |
| M4 — Training & Evaluation | ⬜ Not Started | |

## 📁 Project Structure
```
sri-lankan-food-calorie-estimation/
├── data/
│   ├── raw/                 # Original collected images
│   └── processed/           # Cleaned, resized, augmented images
├── scripts/
│   └── preprocessing/
│       └── preprocess.py    # Quality filter -> resize -> normalise -> augment -> calibrate
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── docs/
│   ├── Milestone2_Report.docx
│   └── architecture_diagram.svg
├── requirements.txt
├── .gitignore
└── README.md
```
> Note: `data/raw/` and `data/processed/` are excluded from version control via
> `.gitignore` because image datasets are large - only the folder structure
> (via `.gitkeep` files) is tracked, not the images themselves.

## ⚙️ Setup
```bash
git clone https://github.com/RashmiDulashani/Sri-Lankan-Food-Calorie-Estimation
cd Sri-Lankan-Food-Calorie-Estimation

python -m venv venv
#Linux/MacOs:
source venv/bin/activate   
# Windows: 
venv\Scripts\activate

pip install -r requirements.txt
```

## 🧹 Running the Preprocessing Pipeline
```bash
python scripts/preprocessing/preprocess.py --input data/raw --output data/processed
```
This applies, in order:
1. **Quality filtering** — Removes blurry, poorly lit, or duplicate images
2. **Resizing** — Standardises all images to 224×224 pixels
3. **Normalisation** — Scales pixel values using ImageNet mean/std
4. **Augmentation** — Random flip, rotation, brightness/contrast (training only)
5. **Reference-object calibration** — Converts pixel measurements to real-world
   scale using a reference object placed beside the plate, for later portion estimation

## 🗂️ Dataset
- **Source:** locally photographed Sri Lankan meals (canteens, home kitchens),
  supplemented by the public Roboflow "SriLankanFoods" set for validation only
- **Target size:** ~3,000–4,000 images across 18–20 dish classes
- **Labels:** multi-label per image (a single plate can contain several dishes)
- **Ethics:** no personally identifiable information collected; consent obtained
  from staff/hosts before photographing meals; see Section 1.3 of the Milestone 2 report

## 🧠 Model Overview
Two-headed architecture on a shared EfficientNet-B0 backbone:
- **Classification head** — multi-label, sigmoid activation, identifies dish(es) on the plate
- **Portion estimation head** — segments food regions and estimates portion size
  using the reference-object calibration from preprocessing
- **Calorie fusion module** — rule-based combination of predicted class(es),
  portion size, and a nutrition reference table to produce the final calorie estimate

Compared against a **ResNet-50 baseline** (single classification head, fixed
average calorie value per class, no portion adjustment) trained and evaluated
under identical conditions.

## 📈 Evaluation Plan
- **Metrics:** macro F1-score, AUC-ROC (classification); MAE, MAPE (calorie estimation)
- **Validation:** stratified 5-fold cross-validation
- **Significance testing:** paired t-test or Wilcoxon signed-rank test on per-fold scores

## 📄 Milestone Documents
| Document | Location |
|---|---|
| Milestone 1 — Research Proposal | `docs/milestone1_research_proposal.pdf` |
| Milestone 2 — Methodology & Data Description | `docs/milestone2_report.pdf` |
| Preprocessing Pipeline| `docs/preprocessing_pipeline.png` |
| System Architecture Diagram | `docs/architecture_diagram.png` |
| Baseline Diagram | `docs/baseline_diagram.png` |
