# 🎮 The Rip-Off Engine
**Fair Market Valuation & Deal Finder for Collectibles using Classical ML**

*Have you ever wondered if that retro console on eBay is a hidden gem or an overpriced scam?* The Rip-Off Engine is an end-to-end Classical Machine Learning pipeline that autonomously predicts the true fair market value of secondary-market collectible games and consoles. By analyzing messy listing titles, item conditions, and seller reputation, it classifies active listings to help buyers instantly spot the best deals.

---

## 🗂️ Project Structure

```
ripoff-engine/
├── data/
│   ├── dataset.csv          # Synthetic dataset (3,000 listings)
│   ├── X_test_raw.pkl       # Holdout test features
│   └── y_test.pkl           # Holdout test targets
├── models/
│   ├── preprocessor.pkl     # Fitted Scikit-Learn ColumnTransformer pipeline
│   └── best_model.pkl       # Tuned Ridge Regression model
├── plots/
│   ├── actual_vs_predicted.png
│   ├── residual_distribution.png
│   └── feature_importances.png
├── features.py              # Custom TitleFeatureExtractor transformer
├── data_gen.py              # Milestone 1 — Synthetic dataset generator
├── pipeline.py              # Milestone 2 — Feature engineering pipeline
├── modeling.py              # Milestone 3 — Model benchmarking & tuning
├── evaluation.py            # Milestone 4 — Residual analysis & plots
├── deal_finder.py           # Interactive CLI tool
└── app.py                   # Cyberpunk Streamlit web dashboard
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo and create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 2. Install dependencies
pip install scikit-learn xgboost pandas numpy matplotlib seaborn joblib streamlit plotly
```

---

## 🚀 Running the Milestones (In Order)

| Step | Script | What it does |
| :--- | :--- | :--- |
| 1 | `python data_gen.py` | Generates `data/dataset.csv` with 3,000 realistic listings |
| 2 | `python pipeline.py` | Builds & saves `models/preprocessor.pkl` |
| 3 | `python modeling.py` | Benchmarks models, tunes best, saves `models/best_model.pkl` |
| 4 | `python evaluation.py` | Generates residual plots in `plots/` |

---

## 🏆 Key Results & Model Leaderboard

During our 5-Fold Cross Validation benchmarking phase, we evaluated three distinct regression architectures.

| Rank | Model | MAE | RMSE | R² |
| :---: | :--- | :--- | :--- | :--- |
| 🥇 | **Ridge Regression** | **$27.20** | — | — |
| 🥈 | Random Forest | $27.95 | — | — |
| 🥉 | XGBoost | $29.93 | — | — |

### Why did Linear Ridge Regression win?
In many modern tabular workflows, tree ensembles (like XGBoost) are assumed to dominate. However, **Ridge Regression (tuned with `alpha=50.0`) outperformed them here.**

**The Rationale:** Our feature space relies heavily on sparse, one-hot encoded categorical variables (specific platforms/conditions) and sparse binary text flags (`has_cib`, `has_box`, etc.). Combined with a moderate sample size (3,000 records) and a high degree of intrinsic market noise, the aggressive L2 regularization of Ridge effectively suppressed extreme weights on rare title keywords — resisting the overfitting that plagued the highly complex, non-linear tree ensembles.

---

## 🔬 The Feature Engineering Breakdown

Transforming messy, capitalized, symbol-ridden marketplace listings into a clean mathematical matrix requires a robust pipeline defined in [`features.py`](features.py) and [`pipeline.py`](pipeline.py):

1. **NLP Text Parsing Flags:** A custom Scikit-Learn transformer (`TitleFeatureExtractor`) scans raw string inputs to extract critical binary indicators such as `has_cib` (Complete in Box), `has_box`, `has_tested`, `has_scratch`, and `has_flaw`.
2. **Categorical & Trust Encoding:** `platform` and `condition` features are strictly One-Hot Encoded. The highly right-skewed `seller_ratings_count` feature is normalized using a continuous `Log1p` transformation.
3. **Strict Anti-Leakage:** Using Scikit-Learn's `ColumnTransformer(remainder='drop')`, we deliberately discard the actual `listed_price` from the training features. This guarantees the model learns to predict the item's inherent value objectively.

---

## ⚖️ Deal Classification Logic

Once the model predicts `true_fair_value`, we compare it against the seller's `listed_price`:

| Verdict | Condition | Indicator |
| :--- | :--- | :--- |
| 🔥 **STEAL** | Listed price ≤ 80% of fair value | ≥ 20% below market |
| ⚖️ **FAIR** | Listed price within ±20% of fair value | At market rate |
| 🛑 **RIP-OFF** | Listed price ≥ 120% of fair value | ≥ 20% above market |

---

## 📊 Visualizations & Residual Analysis

The `evaluation.py` module generates the following analytics on unseen holdout data:

- **`plots/actual_vs_predicted.png`** — Scatter plot of predicted vs. true fair values with a perfect 1:1 benchmark diagonal.
- **`plots/residual_distribution.png`** — KDE histogram of prediction errors to diagnose bias or heteroscedasticity.
- **`plots/feature_importances.png`** — Top-15 feature importances (for tree-based models).

```bash
python evaluation.py
```

---

## 💻 Interactive Interfaces

### 1. 🎮 Cyberpunk Web Dashboard (Streamlit)

A fully immersive, cyberpunk-inspired Fintech web dashboard powered by Streamlit and Plotly. Built in [`app.py`](app.py).

**Features:**
- 🌑 **Glassmorphism UI** — Deep obsidian backgrounds with glowing neon components, subtle grid lines, and `backdrop-filter: blur` glass cards.
- ⚡ **Real-time Inference** — Adjust sliders and inputs; the Ridge model runs instantly on every submission.
- 🎯 **Pulsing Verdict Banner** — STEAL glows neon emerald, RIP-OFF pulses cyber crimson, FAIR shines electric blue — all via CSS keyframe animations.
- 📡 **Plotly Speedometer Gauge** — Dynamic tachometer showing deal severity against the fair-market baseline needle.
- 🏷️ **Neural Keyword Badges** — Positive flags (`CIB`, `TESTED`) glow mint-green; negative flags (`SCRATCH`, `UNTESTED`) glow neon-rose; absent flags render as dimmed inactive slots.
- 🃏 **Quick-Load Presets** — One-click "⚡ STEAL DEAL" and "💀 SCALPER" example buttons.

**To launch:**
```bash
streamlit run app.py
```

---

### 2. 🖥️ Live Deal Finder CLI

For terminal power-users — evaluate any real-world listing on the fly with a color-coded terminal readout.

```bash
python deal_finder.py
```

Enter the listing title, platform, condition, seller feedback, and asking price when prompted. The engine returns a color-coded verdict directly in your terminal.
