# ============================================================
# app.py
# Part 1A
# Smart Streamlit EDA Application
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype
)

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="🧠 Smart Dataset Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Custom Styling
# ------------------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div[data-testid="metric-container"]{
    background:#F8F9FA;
    border-radius:12px;
    padding:12px;
    border:1px solid #EAEAEA;
}

h1,h2,h3{
    color:#0F4C81;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

st.title("🧠 Smart Automated EDA")
st.subheader("📊 Upload any dataset and let the app explore it automatically!")

st.markdown("""
This application automatically:

✅ Detects variable types

✅ Summarises your data

✅ Detects missing values

✅ Detects duplicates

✅ Infers the machine learning problem

✅ Produces interactive visualisations

🚀 Works with CSV and Excel files.
""")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.header("⚙️ Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

# ------------------------------------------------------------
# Cached Data Loader
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(file):

    if file.name.endswith(".csv"):
        return pd.read_csv(file)

    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)

    else:
        raise ValueError("Unsupported file format.")

# ------------------------------------------------------------
# No File Uploaded
# ------------------------------------------------------------

if uploaded_file is None:

    st.info("📁 Upload a dataset to begin.")

    st.stop()

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

try:

    with st.spinner("📊 Loading dataset..."):

        df = load_data(uploaded_file)

except Exception as e:

    st.error(f"❌ Could not read file.\n\n{e}")

    st.stop()

# ------------------------------------------------------------
# Empty Dataset Check
# ------------------------------------------------------------

if df.empty:

    st.error("The uploaded dataset is empty.")

    st.stop()

# ------------------------------------------------------------
# Success Message
# ------------------------------------------------------------

st.success("✅ Dataset loaded successfully!")

# ------------------------------------------------------------
# Detect Datetime Columns Automatically
# ------------------------------------------------------------

for column in df.columns:

    if df[column].dtype == object:

        try:
            converted = pd.to_datetime(df[column])

            if converted.notna().sum() > 0.8 * len(df):

                df[column] = converted

        except:
            pass

# ------------------------------------------------------------
# Detect Variable Types
# ------------------------------------------------------------

numeric_columns = [
    col for col in df.columns
    if is_numeric_dtype(df[col])
]

categorical_columns = [
    col for col in df.columns
    if (
        not is_numeric_dtype(df[col])
        and
        not is_datetime64_any_dtype(df[col])
        and
        not is_bool_dtype(df[col])
    )
]

datetime_columns = [
    col for col in df.columns
    if is_datetime64_any_dtype(df[col])
]

boolean_columns = [
    col for col in df.columns
    if is_bool_dtype(df[col])
]

# ------------------------------------------------------------
# Missing & Duplicate Statistics
# ------------------------------------------------------------

missing_values = int(df.isna().sum().sum())

duplicate_rows = int(df.duplicated().sum())

# ------------------------------------------------------------
# Dataset Metrics
# ------------------------------------------------------------

st.header("📈 Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{df.shape[0]:,}")

c2.metric("Columns", f"{df.shape[1]}")

c3.metric("Missing Values", f"{missing_values:,}")

c4.metric("Duplicate Rows", f"{duplicate_rows:,}")

# ------------------------------------------------------------
# Dataset Preview
# ------------------------------------------------------------

with st.expander("👀 Preview Dataset", expanded=True):

    st.dataframe(df.head())

# ------------------------------------------------------------
# Variable Summary
# ------------------------------------------------------------

with st.expander("📋 Variable Types", expanded=True):

    st.write(f"🔢 Numerical Columns ({len(numeric_columns)})")
    st.write(numeric_columns)

    st.write(f"📝 Categorical Columns ({len(categorical_columns)})")
    st.write(categorical_columns)

    st.write(f"📅 Date Columns ({len(datetime_columns)})")
    st.write(datetime_columns)

    st.write(f"✅ Boolean Columns ({len(boolean_columns)})")
    st.write(boolean_columns)

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

overview_tab, eda_tab, relationships_tab, reports_tab = st.tabs(
    [
        "📊 Overview",
        "📈 EDA",
        "🔗 Relationships",
        "🧠 Reports"
    ]
)

with overview_tab:

    st.subheader("Dataset Overview")

    st.dataframe(df.describe(include="all").T)

with eda_tab:

    st.info("EDA visualisations will be added in Part 1B.")

with relationships_tab:

    st.info("Relationship plots will be added in Part 1B.")

with reports_tab:

    st.info("Automatic reporting features will be added later.")

    # ============================================================
# PART 1B
# Target Selection & Smart EDA
# ============================================================

# ------------------------------------------------------------
# Target Variable Selection
# ------------------------------------------------------------

st.sidebar.header("🎯 Target Variable")

target_column = st.sidebar.selectbox(
    "Select the target variable",
    options=df.columns
)

# ------------------------------------------------------------
# Infer Machine Learning Problem
# ------------------------------------------------------------

target = df[target_column]

problem_type = "Unknown"

if is_numeric_dtype(target):

    unique_values = target.nunique()

    if unique_values == 2:
        problem_type = "Binary Classification"

    elif unique_values <= 15:
        problem_type = "Multi-class Classification"

    else:
        problem_type = "Regression"

else:

    unique_values = target.nunique()

    if unique_values == 2:
        problem_type = "Binary Classification"

    else:
        problem_type = "Multi-class Classification"

st.sidebar.success(f"🧠 Detected Problem: {problem_type}")

# ------------------------------------------------------------
# Missing Value Report
# ------------------------------------------------------------

with overview_tab:

    st.subheader("Missing Values")

    missing_report = (
        df.isna()
        .sum()
        .reset_index()
    )

    missing_report.columns = [
        "Column",
        "Missing Values"
    ]

    missing_report["Percent"] = (
        missing_report["Missing Values"]
        / len(df)
        * 100
    ).round(2)

    st.dataframe(
        missing_report.sort_values(
            "Missing Values",
            ascending=False
        )
    )

# ------------------------------------------------------------
# Duplicate Report
# ------------------------------------------------------------

with overview_tab:

    st.subheader("Duplicate Rows")

    st.metric(
        "Duplicates",
        duplicate_rows
    )

# ============================================================
# NUMERICAL EDA
# ============================================================

with eda_tab:

    st.header("🔢 Numerical Variables")

    if len(numeric_columns) == 0:

        st.warning("No numerical variables detected.")

    else:

        selected_num = st.selectbox(
            "Choose Numerical Variable",
            numeric_columns
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(7,4))

            sns.histplot(
                df[selected_num].dropna(),
                kde=True,
                ax=ax
            )

            ax.set_title(f"Histogram of {selected_num}")

            st.pyplot(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7,4))

            sns.boxplot(
                x=df[selected_num],
                ax=ax
            )

            ax.set_title(f"Boxplot of {selected_num}")

            st.pyplot(fig)

# ------------------------------------------------------------
# Density Plot
# ------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(8,4))

        sns.kdeplot(
            df[selected_num].dropna(),
            fill=True,
            ax=ax
        )

        ax.set_title("Density Plot")

        st.pyplot(fig)

# ------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------

        st.subheader("Summary Statistics")

        st.dataframe(
            df[numeric_columns].describe().T
        )

# ============================================================
# CATEGORICAL EDA
# ============================================================

with eda_tab:

    st.header("📝 Categorical Variables")

    if len(categorical_columns) == 0:

        st.info("No categorical variables detected.")

    else:

        selected_cat = st.selectbox(
            "Choose Categorical Variable",
            categorical_columns
        )

        frequencies = (
            df[selected_cat]
            .value_counts(dropna=False)
        )

        st.dataframe(frequencies)

        fig, ax = plt.subplots(figsize=(8,4))

        sns.countplot(
            data=df,
            x=selected_cat,
            order=df[selected_cat].value_counts().index,
            ax=ax
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)

# ============================================================
# CORRELATION
# ============================================================

with relationships_tab:

    st.header("📈 Correlation Analysis")

    if len(numeric_columns) < 2:

        st.info(
            "Need at least two numerical variables."
        )

    else:

        corr = df[numeric_columns].corr(
            numeric_only=True
        )

        st.dataframe(corr)

        fig, ax = plt.subplots(figsize=(9,7))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)

# ============================================================
# Target Relationships
# ============================================================

with relationships_tab:

    st.header("🎯 Relationship with Target")

    if (
        target_column in numeric_columns
        and
        len(numeric_columns) > 1
    ):

        predictor = st.selectbox(
            "Choose Predictor",
            [
                c
                for c in numeric_columns
                if c != target_column
            ]
        )

        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(
            data=df,
            x=predictor,
            y=target_column,
            ax=ax
        )

        st.pyplot(fig)

    else:

        st.info(
            "Scatter plots require a numeric target."
        )
        # ============================================================
# PART 1C
# Advanced EDA & Smart Visualisations
# ============================================================

# ------------------------------------------------------------
# Missing Value Heatmap
# ------------------------------------------------------------

with relationships_tab:

    st.header("🩹 Missing Value Analysis")

    if missing_values == 0:

        st.success("🎉 No missing values detected!")

    else:

        fig, ax = plt.subplots(figsize=(12,6))

        sns.heatmap(
            df.isnull(),
            cbar=False,
            cmap="viridis",
            ax=ax
        )

        ax.set_title("Missing Value Heatmap")

        st.pyplot(fig)

# ------------------------------------------------------------
# Percentage Missing Plot
# ------------------------------------------------------------

        missing_percent = (
            df.isnull()
            .mean()
            .sort_values(ascending=False)
            * 100
        )

        fig, ax = plt.subplots(figsize=(10,5))

        missing_percent.plot.bar(ax=ax)

        ax.set_ylabel("Percent Missing")

        st.pyplot(fig)

# ============================================================
# IQR OUTLIER DETECTION
# ============================================================

with relationships_tab:

    st.header("🚨 Outlier Detection")

    if len(numeric_columns) == 0:

        st.info("No numeric variables.")

    else:

        outlier_variable = st.selectbox(
            "Variable for Outlier Detection",
            numeric_columns,
            key="outlier"
        )

        Q1 = df[outlier_variable].quantile(.25)
        Q3 = df[outlier_variable].quantile(.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[outlier_variable] < lower)
            |
            (df[outlier_variable] > upper)
        ]

        st.metric(
            "Detected Outliers",
            len(outliers)
        )

        fig, ax = plt.subplots(figsize=(8,4))

        sns.boxplot(
            x=df[outlier_variable],
            ax=ax
        )

        st.pyplot(fig)

        if len(outliers):

            st.dataframe(outliers.head(20))

# ============================================================
# PAIRPLOT
# ============================================================

with reports_tab:

    st.header("📊 Pair Plot")

    if len(numeric_columns) < 2:

        st.info("Need at least two numerical variables.")

    elif len(numeric_columns) > 6:

        st.info(
            "Dataset contains many numerical variables.\n"
            "Displaying the first six."
        )

        pair_columns = numeric_columns[:6]

        fig = sns.pairplot(
            df[pair_columns].dropna()
        )

        st.pyplot(fig)

    else:

        fig = sns.pairplot(
            df[numeric_columns].dropna()
        )

        st.pyplot(fig)

# ============================================================
# DATE ANALYSIS
# ============================================================

with reports_tab:

    st.header("📅 Date Analysis")

    if len(datetime_columns) == 0:

        st.info("No datetime variables detected.")

    else:

        selected_date = st.selectbox(
            "Date Column",
            datetime_columns
        )

        st.write(
            f"Date Range: "
            f"{df[selected_date].min()} "
            f"to "
            f"{df[selected_date].max()}"
        )

        if len(numeric_columns):

            selected_numeric = st.selectbox(
                "Numeric Variable",
                numeric_columns,
                key="dateplot"
            )

            fig, ax = plt.subplots(figsize=(12,5))

            ax.plot(
                df[selected_date],
                df[selected_numeric]
            )

            ax.set_title(
                f"{selected_numeric} over Time"
            )

            st.pyplot(fig)

# ============================================================
# DOWNLOAD DATASET
# ============================================================

with reports_tab:

    st.header("💾 Download Clean Dataset")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download CSV",
        csv,
        file_name="clean_dataset.csv",
        mime="text/csv"
    )

# ============================================================
# DATA QUALITY SCORE
# ============================================================

with overview_tab:

    st.header("🧠 Data Quality Score")

    score = 100

    score -= (
        df.isnull().sum().sum()
        /
        max(df.size,1)
    ) * 50

    score -= (
        duplicate_rows
        /
        max(len(df),1)
    ) * 50

    score = max(0, round(score))

    st.metric(
        "Overall Score",
        f"{score}/100"
    )

    if score >= 90:

        st.success("Excellent quality dataset.")

    elif score >= 75:

        st.info("Good quality dataset.")

    elif score >= 50:

        st.warning("Dataset could benefit from cleaning.")

    else:

        st.error("Significant cleaning recommended.")

# ============================================================
# APPLICATION COMPLETE
# ============================================================

st.success("🎉 Analysis complete!")

st.caption(
    "🧠 Smart Dataset Explorer | "
    "Built with Streamlit, Pandas, Matplotlib & Seaborn"
)
