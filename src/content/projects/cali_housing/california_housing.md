---
title: "California Housing Price Prediction"
description: "Iterative ML on the California Housing dataset—baseline regression to tuned Random Forest with feature selection and model interpretability."
publishDate: "2025-11-04"
tags: ['Machine Learning', 'Data Science', 'XGBoost', 'PyTorch', 'Feature Engineering']
heroImage: "/public/files/images/projects/california_housing/hero.png"
repoUrl: "https://github.com/imranlabs/California_Housing_Price_Prediction/tree/main"
notebookUrl: <a href="https://colab.research.google.com/github/imranlabs/California_Housing_Price_Prediction/blob/main/California_housing_price.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
modelUrl: "/files/cali_housing_best_xgb_model.joblib"
---
## Overview
This project implements a rigorous, end-to-end machine learning workflow to predict median house values in California census block groups. The primary goal was to systematically compare the performance of diverse modeling techniques—from classical regression to advanced boosting and deep learning—while demonstrating proficiency in **feature engineering**, **feature selection**, and **model optimization**.

**Tech Stack:** Python, Scikit-Learn, NumPy, Pandas, Matplotlib, Pytorch   
**Key Skills:** Regression, Feature Engineering, Random Forests, XGBoost, Feature Importance, Pipeline Saving

---

## Why this project
Tabular problems dominate many real-world use cases. My goal was to demonstrate an **end-to-end, reproducible ML workflow**: from baseline → iteration → validation → interpretation → **production-style pipeline**.

---


Methodology and Optimization Pipeline

The project followed an iterative, three-phase approach, prioritizing performance gains and model efficiency:

#### 1. Baseline and Initial Strategy
- **Baseline Models:** Standard **Linear Regression** and initial **Decision Tree** models established the performance floor.
- **Initial Feature Engineering:** An attempt to create new, ratio-based features (e.g., `bedrooms_per_room`) was performed. This step was crucial as it demonstrated that the engineered features **did not improve performance** on tree models, leading to a pivot in strategy.

#### 2. Strategic Feature Selection and Ensemble Learning
Recognizing the limitations of manual feature engineering, the focus shifted to model-driven optimization:

* **Feature Importance:** A Decision Tree model was used to rank feature importance, leading to the elimination of low-impact variables. This strategic **feature selection** reduced the feature set from 11 to **7**, resulting in a slight performance gain and a more efficient model.
* **Ensemble Power (Random Forest):** The optimized 7-feature set was used to train a **Random Forest Regressor**, which delivered the first major performance leap.
* **Advanced Boosting (XGBoost):** An **XGBoost Regressor** was applied with aggressive hyperparameter tuning. This boosting algorithm achieved the project's highest accuracy on the initial feature set.

#### 3. Advanced Feature Engineering & Deep Learning Benchmark
* **Location Clustering (K-Means):** Recognizing that `Latitude` and `Longitude` were top predictors, **K-Means Clustering** was applied to the coordinates to create a new **categorical region feature**. Re-training the XGBoost model on this enhanced feature set delivered the final, best-performing result.
* **Deep Learning Benchmark (PyTorch MLP):** A **Multi-Layer Perceptron (MLP)** was implemented using PyTorch. This step demonstrated proficiency with deep learning frameworks, data preparation for neural networks (scaling, custom data loaders), and benchmarking, even though the XGBoost model maintained superior performance.

---

### Comparative Results Summary

The iterative process led to a significant 28.6% reduction in prediction error (RMSE) compared to the initial Decision Tree model, with the XGBoost model achieving the best overall score.

| Model | Feature Count | RMSE | $R^{2}$ Score | $\Delta$ Improvement (RMSE Reduction) |
| :--- | :--- | :--- | :--- | :--- |
| Decision Tree (Baseline) | 11 | $0.6096$ | $0.7194$ | Baseline |
| Random Forest (Selected) | 7 | $0.5020$ | $0.8090$ | $+17.7\%$ |
| XGBoost (Selected) | 7 | $ 0.4420 $ | $0.8520$ | $+27.5\%$ |
| **XGBoost (with Clustering)** | **$7 + 14$** | **$0.4355$** | **$0.8567$** | **$+28.6\%$** |
| **PyTorch MLP Regressor** | **$7 + 14$** | $0.6630$ | $0.6680$ | - |

> Final Model Performance: The optimized XGBoost Regressor achieved a stellar $R^{2}$ score of $\mathbf{0.8567}$, explaining over $85\%$ of the variance in home prices. Test: The sum is $$\sum_{i=1}^N x_i$$. And inline is $E=mc^2$.
---

### Technical Stack and Takeaways

| Category | Tools & Techniques |
| :--- | :--- |
| **Classical ML** | Scikit-learn, Random Forest, XGBoost, Linear Regression, Decision Tree. |
| **Feature Engineering** | Feature Importance analysis, Feature Selection, **K-Means Clustering** for spatial feature creation. |
| **Deep Learning** | **PyTorch** (custom `Dataset`, `DataLoader`, `nn.Module` definition), $\text{nn.MSELoss()}$, $\text{Adam}$ optimizer, GPU/MPS device optimization. |
| **Data & Metrics** | Pandas, NumPy, RMSE, $R^2$ Score. |
| **Model Persistence** | Final XGBoost model was saved using `joblib` for deployment. |

**Key Takeaways for Portfolio:**
1.  **Iterative Optimization:** Demonstrated a principled approach by using initial models to guide later, more complex steps (Feature Importance over failed Feature Engineering).
2.  **Handling Tabular Data:** Confirmed that ensemble and boosting methods (XGBoost) are often superior to simple Deep Learning architectures for structured, tabular data.
3.  **Cross-Platform Proficiency:** Successfully implemented and benchmarked models in both the **Scikit-learn/XGBoost** ecosystem and the **PyTorch** deep learning framework, demonstrating versatility.

---

## What I learned
- Iterative modeling (baseline → trees → ensembles) combined with **feature importance pruning** yields solid gains on tabular data.  
- **Diagnostic plots** and **permutation importance** add trust and explainability.  
- Packaging everything into a **single Pipeline** simplifies **reproducible inference** and future deployment.

---

## Links & Artifacts
- 📓 Notebook: [Enhanced notebook]({frontmatter.notebookUrl})  
- 💾 Trained model (pipeline): [Download]({frontmatter.modelUrl})  
- 🧑‍💻 Code: [GitHub repo]({frontmatter.repoUrl})

