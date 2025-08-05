# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
import pandas as pd
import mlflow
from mlflow.models import infer_signature


# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
df=pd.read_csv("data.csv")
df.head(10)


# -------------------- Inspection and clean up -----------------
# -------------------- Inspection and clean up -----------------
# -------------------- Inspection and clean up -----------------
import seaborn as sns
print(df.dtypes)
print(df.isnull().any(axis=1).sum()) # null values
print(df.isnull().sum()) # null values
sns.heatmap(df.isnull())

df = df.dropna()
sns.heatmap(df.isnull())

# -------------------- Visualisations -----------------
# -------------------- Visualisations -----------------
# -------------------- Visualisations -----------------
import numpy as np
import warnings
warnings.filterwarnings("ignore")

g = sns.FacetGrid(df)
g.map(sns.scatterplot, 'tenure', 'Churn')

# -------------------- Cleanup -----------------
# -------------------- Cleanup -----------------
# -------------------- Cleanup -----------------

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
df_enc = df.copy()
df_enc['customer_id'] = ord_enc.fit_transform(df[['customerID']])
df_enc['gender_code'] = ord_enc.fit_transform(df[['gender']])
df_enc['partner_code'] = ord_enc.fit_transform(df[['Partner']])
df_enc['dependents_code'] = ord_enc.fit_transform(df[['Dependents']])
df_enc['phone_service_code'] = ord_enc.fit_transform(df[['PhoneService']])
df_enc['multiple_lines_code'] = ord_enc.fit_transform(df[['MultipleLines']])
df_enc['internet_service_code'] = ord_enc.fit_transform(df[['InternetService']])
df_enc['online_security_code'] = ord_enc.fit_transform(df[['OnlineSecurity']])
df_enc['online_backup_code'] = ord_enc.fit_transform(df[['OnlineBackup']])
df_enc['device_protection_code'] = ord_enc.fit_transform(df[['DeviceProtection']])
df_enc['tech_support_code'] = ord_enc.fit_transform(df[['TechSupport']])
df_enc['streaming_tv_code'] = ord_enc.fit_transform(df[['StreamingTV']])
df_enc['streaming_movies_code'] = ord_enc.fit_transform(df[['StreamingMovies']])
df_enc['contract_code'] = ord_enc.fit_transform(df[['Contract']])
df_enc['paperless_billing_code'] = ord_enc.fit_transform(df[['PaperlessBilling']])
df_enc['payment_method_code'] = ord_enc.fit_transform(df[['PaymentMethod']])
df_enc['churn_num'] = ord_enc.fit_transform(df[['Churn']])
df_enc['total_charges_num'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df_enc.isnull().sum()
df_enc_sel = df_enc[['customer_id','gender_code','partner_code','dependents_code', 'phone_service_code', 'multiple_lines_code', 'internet_service_code',
                     'online_security_code', 'online_backup_code', 'device_protection_code', 'tech_support_code', 'streaming_tv_code',
                     'streaming_movies_code','contract_code','paperless_billing_code','payment_method_code',
                     'SeniorCitizen','tenure','MonthlyCharges','total_charges_num','churn_num']]


# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------

df_enc_sel.corr()['churn_num'].sort_values(ascending = False).plot(kind='bar')
df_enc_sel = df_enc_sel.dropna()
df_enc_sel.isnull().sum()


# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
import matplotlib.pyplot as plt
plt.matshow(df_enc_sel.corr())

# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
corr = df_enc_sel.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
# ----------------- correlation visualization -----------------
numerical_cols=df_enc_sel.select_dtypes(include=np.number).columns.tolist()
df_enc_sel_cols=df_enc_sel[numerical_cols].columns.difference(['customer_id'])
sns.pairplot(df_enc_sel[['MonthlyCharges', 'SeniorCitizen', 'contract_code', 'dependents_code',
                         'device_protection_code', 'gender_code', 'internet_service_code',
                         'multiple_lines_code', 'online_backup_code', 'online_security_code',
                         'paperless_billing_code', 'partner_code', 'payment_method_code',
                         'phone_service_code', 'streaming_movies_code', 'streaming_tv_code',
                         'tech_support_code', 'tenure', 'total_charges_num', 'churn_num']], hue = 'churn_num')

# ----------------- modeling -----------------
# ----------------- modeling -----------------
# ----------------- modeling -----------------
# class imbalance fix using SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
sm = SMOTE(random_state=0, sampling_strategy=1.0)

X=df_enc_sel[df_enc_sel[df_enc_sel.select_dtypes(include=np.number).columns.tolist()].columns.difference(['customer_id','churn_num'])]
y=df_enc_sel['churn_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)


# ----------------- modeling - Logistic regression -----------------
# ----------------- modeling - Logistic regression -----------------
# ----------------- modeling - Logistic regression -----------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
params_lr = {
    "random_state":13
}
logreg_smote = LogisticRegression(**params_lr)
logreg_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = logreg_smote.predict(X_test)
logit_roc_auc_smote = roc_auc_score(y_test, y_pred_smote)
print(classification_report(y_test, y_pred_smote))
print(f"The area under the curve is: {logit_roc_auc_smote}")

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Telecom Churn prediction")
with mlflow.start_run(run_name="Logistic Regression"):
    mlflow.log_params(params_lr)
    mlflow.log_metric("area_under_curve", logit_roc_auc_smote)
    mlflow.set_tag("Training Info", "Basic Logistic regression")
    signature = infer_signature(X_train, logreg_smote.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        sk_model=logreg_smote,
        artifact_path="logreg_smote",
        signature=signature,
        input_example=X_train,
        registered_model_name="logreg_smote",
    )

# ----------------- modeling - Decision Tree -----------------
# ----------------- modeling - Decision Tree -----------------
# ----------------- modeling - Decision Tree -----------------
from sklearn.tree import DecisionTreeClassifier
params_dt = {
    "random_state":13,
    'criterion': 'entropy'
}
dectree = DecisionTreeClassifier(**params_dt)
dectree.fit(X_train_smote, y_train_smote)
y_pred_dectree = dectree.predict(X_test)
dectree_roc_auc = roc_auc_score(y_test, y_pred_dectree)
print(classification_report(y_test, y_pred_dectree))
print(f"The area under the curve is: {dectree_roc_auc}")

with mlflow.start_run(run_name="Decision Tree"):
    mlflow.log_params(params_dt)
    mlflow.log_metric("area_under_curve", dectree_roc_auc)
    mlflow.set_tag("Training Info", "Decision Tree Classifier")
    signature = infer_signature(X_train, dectree.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        sk_model=dectree,
        artifact_path="dectree",
        signature=signature,
        input_example=X_train,
        registered_model_name="dectree",
    )

# ----------------- modeling - Random forest -----------------
# ----------------- modeling - Random forest -----------------
# ----------------- modeling - Random forest -----------------
from sklearn.ensemble import RandomForestClassifier
params_rf = {
    "max_depth":5
}
randomforest = RandomForestClassifier(**params_rf)
randomforest.fit(X_train_smote, y_train_smote)
y_pred_randomforest = randomforest.predict(X_test)
randomforest_roc_auc = roc_auc_score(y_test, y_pred_randomforest)
print(classification_report(y_test, y_pred_randomforest))
print(f"The area under the curve is: {randomforest_roc_auc}")
y_pred_probabilities_random_forest = randomforest.predict_proba(X_test)

with mlflow.start_run(run_name="Random Forest"):
    mlflow.log_params(params_rf)
    mlflow.log_metric("area_under_curve", randomforest_roc_auc)
    mlflow.set_tag("Training Info", "Random Forest Classifier")
    signature = infer_signature(X_train, randomforest.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        sk_model=randomforest,
        artifact_path="randomforest",
        signature=signature,
        input_example=X_train,
        registered_model_name="randomforest",
    )

# ----------------- modeling - Ada Boost -----------------
# ----------------- modeling - Ada Boost -----------------
# ----------------- modeling - Ada Boost -----------------
from sklearn.ensemble import AdaBoostClassifier
params_ab = {
    "n_estimators": 100
}
adaboost = AdaBoostClassifier(**params_ab)
adaboost.fit(X_train_smote, y_train_smote)
y_pred_adaboost = adaboost.predict(X_test)
adaboost_roc_auc = roc_auc_score(y_test, y_pred_adaboost)
print(classification_report(y_test, y_pred_adaboost))
print(f"The area under the curve is: {adaboost_roc_auc}")
y_pred_probabilities_adaboost = adaboost.predict_proba(X_test)

with mlflow.start_run(run_name="AdaBoost Classifier"):
    mlflow.log_params(params_ab)
    mlflow.log_metric("area_under_curve", adaboost_roc_auc)
    mlflow.set_tag("Training Info", "Ada Boost Classifier")
    signature = infer_signature(X_train, adaboost.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        sk_model=adaboost,
        artifact_path="adaboost",
        signature=signature,
        input_example=X_train,
        registered_model_name="adaboost",
    )


# ----------------- modeling - Gradient Boost -----------------
# ----------------- modeling - Gradient Boost -----------------
# ----------------- modeling - Gradient Boost -----------------
from sklearn.ensemble import GradientBoostingClassifier
gradientboost = GradientBoostingClassifier()
gradientboost.fit(X_train_smote, y_train_smote)
y_pred_gradientboost = gradientboost.predict(X_test)
gradientboost_roc_auc = roc_auc_score(y_test, y_pred_gradientboost)
print(classification_report(y_test, y_pred_gradientboost))
print(f"The area under the curve is: {gradientboost_roc_auc}")
y_pred_probabilities_gradientboost = gradientboost.predict_proba(X_test)
params_gb = gradientboost.get_params()


with mlflow.start_run():
    mlflow.log_params(params_gb)
    mlflow.log_metric("area_under_curve", gradientboost_roc_auc)
    mlflow.set_tag("Training Info", "Ada Boost Classifier")
    signature = infer_signature(X_train, gradientboost.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        sk_model=gradientboost,
        artifact_path="gradientboost",
        signature=signature,
        input_example=X_train,
        registered_model_name="gradientboost",
    )

# ----------------- Feature Importance -----------------
# ----------------- Feature Importance -----------------
# ----------------- Feature Importance -----------------


def feature_importance(model, model_name):
    feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    feature_importances = feature_importances.sort_values(axis=0, ascending=False)
    plt.figure(figsize=(15,15))
    fig, ax = plt.subplots()
    feature_importances.plot.bar()
    ax.set_title("Feature importances")
    fig.tight_layout()
    plt.savefig(f'tree_feature_importance_{model_name}.png')

feature_importance(dectree, "dectree")
feature_importance(randomforest, "randomforest")
feature_importance(adaboost, "adaboost")
feature_importance(gradientboost, "gradientboost")


# ----------------- LIME Implementation -----------------
# ----------------- LIME Implementation -----------------
# ----------------- LIME Implementation -----------------
import lime.lime_tabular
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_smote.values,
    feature_names = X_train_smote.columns,
    class_names=['Not Churn','Churn'],
    kernel_width=5)
choosen_instance = X_test.loc[[1]].values[0]
exp = explainer.explain_instance(
    choosen_instance,
    lambda x: gradientboost.predict_proba(x).astype(float),
    num_features=10)
exp.show_in_notebook(show_all=False)
exp.as_list()

import mlflow
from mlflow.models import infer_signature
def ml_flow():
    with mlflow.start_run():
        mlflow.log_params(params_rf)
        mlflow.log_metric("area_under_curve", randomforest_roc_auc)
        mlflow.set_tag("Training Info", "Random Forest Classifier")
        signature = infer_signature(X_train, randomforest.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            sk_model=randomforest,
            artifact_path="randomforest",
            signature=signature,
            input_example=X_train,
            registered_model_name="randomforest",
        )