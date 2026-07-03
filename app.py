# ============================================================
# 🧠 Universal EDA + ML Workbench (Streamlit App)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# ============================================================
# Streamlit Config
# ============================================================

st.set_page_config(
    page_title="🧠 Universal ML Workbench",
    page_icon="📊",
    layout="wide"
)

st.title("🧠 Universal EDA + Machine Learning Workbench")
st.markdown("Upload any dataset and automatically explore + model it.")

# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

st.success("Dataset loaded successfully!")

# ============================================================
# Quick EDA
# ============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isna().sum().sum())
col4.metric("Duplicate Rows", df.duplicated().sum())

st.write("### Data Preview")
st.dataframe(df.head())

st.write("### Data Types")
st.write(df.dtypes)

st.write("### Missing Values per Column")
st.write(df.isna().sum())

# ============================================================
# Feature Detection
# ============================================================

def detect_types(dataframe):
    numeric = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical = dataframe.select_dtypes(include="object").columns.tolist()
    return numeric, categorical

num_cols, cat_cols = detect_types(df)

st.subheader("🧠 Detected Feature Types")

st.write("**Numerical Columns:**")
st.write(num_cols)

st.write("**Categorical Columns:**")
st.write(cat_cols)

# ============================================================
# Target Selection
# ============================================================

st.subheader("🎯 Target Selection")

target = st.selectbox("Select target column", df.columns)

X = df.drop(columns=[target])
y = df[target]

def detect_task(y):
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"

task = detect_task(y)

st.info(f"Detected task type: **{task.upper()}**")

# ============================================================
# Preprocessing
# ============================================================

num_features = X.select_dtypes(include=np.number).columns
cat_features = X.select_dtypes(include="object").columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# Model Training
# ============================================================

if task == "classification":
    model = KNeighborsClassifier(n_neighbors=5)
else:
    model = KNeighborsRegressor(n_neighbors=5)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

st.subheader("📈 Model Evaluation")

if task == "classification":

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Precision", f"{prec:.3f}")
    col3.metric("Recall", f"{rec:.3f}")
    col4.metric("F1 Score", f"{f1:.3f}")

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

else:

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("MAE", f"{mae:.3f}")
    col3.metric("R²", f"{r2:.3f}")

    fig = px.scatter(
        x=y_test,
        y=y_pred,
        title="Actual vs Predicted"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Learning Curve
# ============================================================

st.subheader("📉 Learning Curve")

scoring = "accuracy" if task == "classification" else "r2"

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=3,
    scoring=scoring,
    n_jobs=-1
)

fig, ax = plt.subplots()

ax.plot(train_sizes, train_scores.mean(axis=1), label="Training Score")
ax.plot(train_sizes, test_scores.mean(axis=1), label="Validation Score")

ax.set_title("Learning Curve")
ax.set_xlabel("Training Samples")
ax.set_ylabel("Score")
ax.legend()

st.pyplot(fig)
