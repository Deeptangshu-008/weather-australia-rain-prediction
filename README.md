# 🌦️ Australia Weather Rain Prediction

A machine learning classification project that predicts whether it will rain the next day in Australia using historical weather observations.

The project follows a complete Scikit-Learn machine learning workflow, including data cleaning, missing-value analysis, feature engineering, preprocessing pipelines, baseline model comparison, hyperparameter tuning, overfitting analysis, and final evaluation on a completely unseen future year.

---

## 📌 Project Overview

The objective of this project is to predict the target variable:

**`RainTomorrow`**

- `0` → No rain tomorrow
- `1` → Rain tomorrow

Instead of randomly splitting the dataset, the data was divided chronologically to simulate a realistic forecasting scenario where the model learns from the past and predicts the future.

### Dataset

- **145,460 observations**
- **23 columns**
- Multiple weather stations across Australia
- Approximately 10 years of daily observations
- Target: `RainTomorrow`

---

## 🎯 Project Objectives

The main objectives of this project were:

- Understand and clean real-world weather data
- Analyze missing values and their relationship with the target
- Perform feature engineering
- Build a reusable preprocessing pipeline
- Compare multiple classification algorithms
- Evaluate models using multiple classification metrics
- Tune the best-performing model
- Analyze and reduce overfitting
- Evaluate the final model on a completely unseen future year

---

## 🧹 Data Preprocessing

The dataset contains a significant amount of missing data in several weather-related features.

Instead of automatically removing columns with high missing percentages, the missingness and usefulness of individual features were investigated.

Features such as:

- `Sunshine`
- `Evaporation`
- `Cloud9am`
- `Cloud3pm`

were retained because they contained useful predictive information.

Numerical missing values were handled using median imputation through the preprocessing pipeline.

Categorical variables were encoded using One-Hot Encoding.

The target variables `RainToday` and `RainTomorrow` were converted from:

```text
No → 0
Yes → 1
```

🧠 Feature Engineering

Several useful features were extracted from the date information and weather observations.

Examples include:

Year
Month
Day
Day of week
Weekend indicators
Other weather-derived features

Date information was handled before model training while learned preprocessing operations such as imputation and encoding were performed through the pipeline.

⏳ Train / Validation / Test Split

Because this is a weather forecasting problem, a chronological split was used instead of a random train-test split.

2007 – 2014 → Training Set
2015 – 2016 → Validation Set
2017 → Test Set

Training : 101,018
Validation : 35,819
Test : 8,623

Rows with missing target values were excluded from evaluation where necessary.

Why chronological splitting?

A random split could allow observations from the future to influence the training process.

The chronological split better represents the real-world situation:
Past Weather Data
↓
Training
↓
Validation
↓
Future Data
↓
2017
Test Set
The 2017 test set was kept untouched during model selection and hyperparameter tuning.

🤖 Models Evaluated

The following classification algorithms were tested:

Decision Tree
Random Forest
AdaBoost
Gradient Boosting
XGBoost

All models were evaluated using the same training and validation framework.

📊 Baseline Model Comparison
The models were initially compared using the validation set before selecting a model for further tuning.
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ----------------- | -------: | --------: | -----: | -------: | ------: |
| Decision Tree | 83.04% | 69.34% | 44.43% | 54.16% | 82.37% |
| Random Forest | 83.49% | 79.65% | 35.93% | 49.52% | 85.04% |
| AdaBoost | 83.91% | 68.35% | 53.31% | 59.90% | 85.87% |
| Gradient Boosting | 85.03% | 73.45% | 52.60% | 61.30% | 87.79% |
| XGBoost | 84.86% | 72.39% | 53.85% | 61.23% | 87.26% |

Model Selection

Gradient Boosting and XGBoost were the strongest baseline models.

XGBoost was selected for further experimentation because of its strong overall performance and its ability to provide extensive regularization and tuning options.

🚀 XGBoost Hyperparameter Tuning

The XGBoost model was tuned by systematically testing different values of important hyperparameters.

Parameters explored included:

max_depth
n_estimators
learning_rate
subsample
min_child_weight
colsample_bytree
reg_alpha
reg_lambda

The goal was not simply to maximize training performance, but to improve validation performance while reducing the amount of overfitting.

Final XGBoost Parameters:
XGBClassifier(
random_state=42,
n_jobs=-1,
max_depth=7,
n_estimators=200,
learning_rate=0.10,
subsample=0.7,
min_child_weight=5,
colsample_bytree=0.9,
reg_alpha=0.1,
reg_lambda=1.5
)

📉 Overfitting Analysis

The original XGBoost model showed significant overfitting.

Before tuning:

Training ROC-AUC = 0.9964
Validation ROC-AUC = 0.8726
Train–Validation Gap = 0.1238

After hyperparameter tuning:
Training ROC-AUC = 0.9376
Validation ROC-AUC = 0.8834
Train–Validation Gap = 0.0542

The tuned model achieved a higher validation ROC-AUC while substantially reducing the train-validation gap.

This indicates improved generalization rather than simply improving performance on the training data.

🏆 Final Model Performance

The tuned XGBoost model was evaluated on the previously untouched 2017 test set.

Validation Performance:
Accuracy : 85.49%
Precision : 75.61%
Recall : 52.68%
F1 Score : 62.64%
ROC-AUC : 88.34%

Final 2017 Test Performance:
Accuracy : 85.85%
Precision : 74.33%
Recall : 48.95%
F1 Score : 59.03%
ROC-AUC : 87.70%

The validation ROC-AUC was 0.8834, while the final 2017 test ROC-AUC was 0.8770.

The relatively small difference suggests that the model generalizes reasonably well to the unseen future year.

🔢 Final Test Confusion Matrix
[[6405  298]
 [ 900  863]]

Where:
Predicted
No Rain Rain

Actual No Rain 6405 298
Actual Rain 900 863

Therefore:

True Negatives (TN): 6405
False Positives (FP): 298
False Negatives (FN): 900
True Positives (TP): 863

The model correctly identified 863 rainy days in the 2017 test set.

When the model predicted rain, its precision was approximately 74.33%.

📈 Visualizations

The project includes visualizations for model evaluation and interpretation.

Confusion Matrix
ROC Curve
Feature Importance
Model Comparison

🔍 Feature Importance

XGBoost feature importance was extracted from the trained model to identify which transformed weather features contributed most to the predictions.

The feature importance analysis provides insight into which weather observations were most useful for predicting rainfall on the following day.

The visualization is available in:
images/feature_importance.png

🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
XGBoost
Jupyter Notebook

📂 Project Structure
weather-australia-rain-prediction/
│
├── data/
│ └── README.md
│
├── images/
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── feature_importance.png
│ └── model_comparison.png
│
├── weather.ipynb
│  
│
├── .gitignore
├── requirements.txt
└── README.md

The raw dataset is not included in the repository.

💡 Key Learnings

This project provided practical experience with:

Real-world data cleaning
Missing-value analysis
Categorical encoding
Numerical feature preprocessing
Scikit-Learn Pipeline
Binary classification
Model comparison
Precision vs Recall trade-offs
ROC-AUC
Confusion matrices
Hyperparameter tuning
Overfitting detection
Chronological train-validation-test splitting
Future-data evaluation
XGBoost
Feature importance

One of the most important lessons from this project was that a model with a very high training score is not necessarily a good model.

The final XGBoost model achieved a lower training ROC-AUC than the original model but performed better on validation data and had a substantially smaller train-validation gap.

📌 Conclusion

The final tuned XGBoost classifier achieved:

87.70% ROC-AUC on the completely unseen 2017 test set.

The model demonstrated reasonably strong generalization from historical weather data to a future year.

The project also demonstrates a complete machine learning workflow rather than focusing only on model accuracy:
Data
↓
Exploration
↓
Cleaning
↓
Feature Engineering
↓
Preprocessing Pipeline
↓
Chronological Split
↓
Baseline Models
↓
Model Comparison
↓
XGBoost Selection
↓
Hyperparameter Tuning
↓
Overfitting Analysis
↓
Final 2017 Test Evaluation

👨‍💻 Author

Deeptangshu Ghosh

B.Tech Computer Science & Engineering

⭐ If you found this project useful, feel free to star the repository.
