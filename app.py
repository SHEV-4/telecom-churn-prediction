import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

feature_order = [
    "is_tv_subscriber",
    "is_movie_package_subscriber",
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
    "download_over_limit",
    "reamining_contract_missing",
]


# Load models
models = {
    "Logistic Regression": joblib.load("logistic.pkl"),
    "Random Forest": joblib.load("random_forest.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting.pkl"),
    "Neural Network (MLP)": joblib.load("mlp.pkl"),
}


# Load metrics & ROC
with open("metrics.json") as f:
    metrics = json.load(f)

with open("roc_data.json") as f:
    roc_data = json.load(f)


if "ready" not in st.session_state:
    st.session_state.ready = False

if "input_df" not in st.session_state:
    st.session_state.input_df = None


# UI
st.set_page_config(page_title="Churn Prediction", layout="wide")

selected = option_menu(
    menu_title=None,
    options=list(models.keys()),
    menu_icon="cast",
    orientation="horizontal",
)

plot_type = st.sidebar.selectbox("Виберіть відображення:", ["Метрики", "ROC-крива"])


# Feature input
st.sidebar.subheader("Дані клієнта")


# Чекбокс: чи відомий контракт
contract_known = st.sidebar.checkbox("Відомий термін контракту?", value=True)

if contract_known:
    reamining_contract = st.sidebar.number_input("Remaining contract (years)", 0.0, 5.0)
    reamining_contract_missing = 0
else:
    reamining_contract = 0.0
    reamining_contract_missing = 1

input_data = {
    "is_tv_subscriber": st.sidebar.selectbox("TV subscriber", [0, 1]),
    "is_movie_package_subscriber": st.sidebar.selectbox("Movie package", [0, 1]),
    "subscription_age": st.sidebar.number_input("Subscription age", 0.0, 20.0),
    "bill_avg": st.sidebar.number_input("Bill avg", 0.0, 500.0),
    "reamining_contract": reamining_contract,
    "service_failure_count": st.sidebar.number_input("Service failures", 0, 20),
    "download_avg": st.sidebar.number_input("Download avg", 0.0, 5000.0),
    "upload_avg": st.sidebar.number_input("Upload avg", 0.0, 500.0),
    "download_over_limit": st.sidebar.selectbox("Over limit", [0, 1]),
    "reamining_contract_missing": reamining_contract_missing,
}

button = st.sidebar.button("Спрогнозувати")


# Prediction
if button:
    st.session_state.input_df = pd.DataFrame(
        [[input_data[col] for col in feature_order]], columns=feature_order
    )
    st.session_state.ready = True
if st.session_state.ready:
    model = models[selected]

    proba = model.predict_proba(st.session_state.input_df)[0][1]
    prediction = int(proba >= 0.5)

    st.subheader("Результат прогнозу")
    st.metric("Ймовірність відтоку", f"{proba:.2%}")

    if proba > 0.7:
        st.error("Високий ризик відтоку")
    elif proba > 0.4:
        st.warning("Середній ризик")
    else:
        st.success("Низький ризик")


# Metrics / ROC display
if plot_type == "Метрики":
        st.subheader(f"Метрики моделі: {selected}")

        m = metrics[selected]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{m['accuracy']:.3f}")
        col2.metric("Precision", f"{m['precision']:.3f}")
        col3.metric("Recall", f"{m['recall']:.3f}")
        col4.metric("F1-score", f"{m['f1_score']:.3f}")

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(m["classification_report"]).T)

if plot_type == "ROC-крива":
        st.subheader(f"ROC Curve — {selected}")

        fpr = roc_data[selected]["fpr"]
        tpr = roc_data[selected]["tpr"]
        auc = roc_data[selected]["auc"]

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()

        st.pyplot(plt)
        plt.close()

