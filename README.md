#  The Rip-Off Engine
**Fair Market Valuation & Deal Finder for Collectibles using Classical ML**

*Have you ever wondered if that retro console on eBay is a hidden gem or an overpriced scam?* The Rip-Off Engine is an end-to-end Classical Machine Learning pipeline that autonomously predicts the true fair market value of secondary-market collectible games and consoles. By analyzing messy listing titles, item conditions, and seller reputation, it classifies active listings to help buyers instantly spot the best deals.

---

##  Key Results & Model Leaderboard

During our 5-Fold Cross Validation benchmarking phase, we evaluated three distinct regression architectures.

| Model | Mean Absolute Error (MAE) |
| :--- | :--- |
| ** Ridge Regression** | **$27.20** |
| Random Forest | $27.95 |
| XGBoost | $29.93 |

### Why did Linear Ridge Regression win?
In many modern tabular workflows, tree ensembles (like XGBoost) are assumed to dominate. However, **Ridge Regression (tuned with `alpha=50.0`) outperformed them here.** 

**The Rationale:** Our feature space relies heavily on sparse, one-hot encoded categorical variables (specific platforms/conditions) and sparse binary text flags (`has_cib`, `has_box`, etc.). Combined with a moderate sample size (3,000 records) and a high degree of intrinsic market noise (anomalous pricing), the aggressive L2 regularization of Ridge effectively suppressed extreme weights on rare title keywords, resisting the overfitting that plagued the highly complex, non-linear tree ensembles.

---

##  The Feature Engineering Breakdown

Transforming messy, capitalized, symbol-ridden marketplace listings into a clean mathematical matrix requires a robust pipeline:

1. **NLP Text Parsing Flags:** A custom Scikit-Learn transformer (`TitleFeatureExtractor`) scans raw string inputs to extract critical binary indicators such as `has_cib` (Complete in Box), `has_box`, `has_tested`, `has_scratch`, and `has_flaw`.
2. **Categorical & Trust Encoding:** `platform` and `condition` features are strictly One-Hot Encoded. The highly right-skewed `seller_ratings_count` feature is normalized using a continuous `Log1p` mathematical transformation.
3. **Strict Anti-Leakage:** Using Scikit-Learn's `ColumnTransformer(remainder='drop')`, we deliberately discard the actual `listed_price` from the training features. This guarantees the model learns to predict the item's inherent value objectively, rather than cheating by referencing the seller's asking price.

---

##  Deal Classification Logic

Once the model predicts the `true_fair_value`, we evaluate it against the actual `listed_price` to calculate the residual variance. The engine classifies the listing into one of three distinct tiers:

*    **STEAL:** Listed $\le -20\%$ below the predicted fair value.
*    **FAIR:** Listed within $\pm 20\%$ of the predicted fair value.
*    **RIP-OFF:** Listed $\ge +20\%$ above the predicted fair value.

---

##  Visualizations & Residual Analysis

The project includes an evaluation module that automatically generates analytics on unseen holdout data. Look in the `plots/` directory for:
- **`plots/actual_vs_predicted.png`:** A scatter plot visualizing the model's predictive variance alongside the perfect 1:1 benchmark diagonal.
- **`plots/residual_distribution.png`:** A KDE histogram displaying the distribution of the prediction errors (Actual Fair Value - Predicted Fair Value), helping to diagnose heteroscedasticity or bias.

*(Run `python evaluation.py` to dynamically generate these artifacts.)*

---

##  Interactive CLI: Live Deal Finder

Evaluate custom, real-world listings on the fly using our interactive terminal tool! 

**To launch the Deal Finder:**
```bash
python deal_finder.py
```

The CLI will prompt you for the listing details (Title, Platform, Condition, Seller Feedback, Listed Price), instantly route the inputs through the pickled Scikit-Learn feature pipeline, run inference using the tuned Ridge Regression model, and print a color-coded verdict (STEAL, FAIR, or RIP-OFF).
