# Telco Customer Churn Prediction

This project predicts customer churn using various ML models and logs experiments using MLflow. It also includes LIME for model explainability.

## 📂 Dataset

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## ⚙️ Models Trained

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting

## 🛠 Tools Used

- `scikit-learn` for modeling
- `imbalanced-learn` for SMOTE
- `MLflow` for experiment tracking
- `LIME` for interpretability
- `matplotlib`, `seaborn` for visualization

## ▶️ How to Run

```bash
pip install -r requirements.txt
python telco_churn_model.py
