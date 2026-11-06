import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Subscription Churn Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv(r"E:\Data Visualisation Projects\StreamlitProject\customer_subscription_churn_analysis.csv")
    data["signup_date"] = pd.to_datetime(data["signup_date"], errors="coerce")
    return data

st.title("📊 Customer Subscription Churn Analysis Dashboard")
st.markdown("This Streamlit project analyzes customer subscription churn using Pandas, NumPy, Matplotlib, and Seaborn.")

st.sidebar.header("Upload / Filter Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
df = load_data(uploaded_file)

st.sidebar.subheader("Filters")
plan_options = sorted(df["plan_type"].dropna().unique().tolist())
selected_plans = st.sidebar.multiselect("Select Plan Type", plan_options, default=plan_options)

churn_options = sorted(df["churn"].dropna().unique().tolist())
selected_churn = st.sidebar.multiselect("Select Churn Status", churn_options, default=churn_options)

min_fee = int(df["monthly_fee"].min())
max_fee = int(df["monthly_fee"].max())
selected_fee = st.sidebar.slider("Monthly Fee Range", min_fee, max_fee, (min_fee, max_fee))

filtered_df = df[
    (df["plan_type"].isin(selected_plans)) &
    (df["churn"].isin(selected_churn)) &
    (df["monthly_fee"].between(selected_fee[0], selected_fee[1]))
]

st.subheader("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", len(filtered_df))
col2.metric("Average Monthly Fee", f"₹{filtered_df['monthly_fee'].mean():.2f}" if len(filtered_df) else "₹0")
col3.metric("Average Tenure", f"{filtered_df['tenure_months'].mean():.1f} months" if len(filtered_df) else "0 months")
churn_rate = (filtered_df["churn"].eq("Yes").mean() * 100) if len(filtered_df) else 0
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(20), use_container_width=True)

st.subheader("Dataset Information")
info_col1, info_col2 = st.columns(2)
with info_col1:
    st.write("Shape of Dataset")
    st.write(filtered_df.shape)
    st.write("Column Names")
    st.write(list(filtered_df.columns))
with info_col2:
    st.write("Missing Values")
    st.dataframe(filtered_df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}))

st.subheader("Summary Statistics")
st.dataframe(filtered_df.describe(include="all"), use_container_width=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Churn Analysis", "Plan Analysis", "Numerical Analysis", "Correlation", "Custom Plot"
])

with tab1:
    st.header("Churn Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=filtered_df, x="churn", ax=ax)
    ax.set_title("Churn Count")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.header("Churn by Plan Type")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=filtered_df, x="plan_type", hue="churn", ax=ax)
    ax.set_title("Plan Type vs Churn")
    ax.set_xlabel("Plan Type")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

with tab2:
    st.header("Plan Type Distribution")
    plan_counts = filtered_df["plan_type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(plan_counts.index, plan_counts.values)
    ax.set_title("Customers by Plan Type")
    ax.set_xlabel("Plan Type")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.header("Average Monthly Fee by Plan Type")
    avg_fee = filtered_df.groupby("plan_type", as_index=False)["monthly_fee"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=avg_fee, x="plan_type", y="monthly_fee", ax=ax)
    ax.set_title("Average Monthly Fee by Plan")
    ax.set_xlabel("Plan Type")
    ax.set_ylabel("Average Monthly Fee")
    st.pyplot(fig)

with tab3:
    st.header("Numerical Column Analysis")
    numeric_columns = filtered_df.select_dtypes(include=np.number).columns.tolist()
    selected_numeric = st.selectbox("Select Numeric Column", numeric_columns)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(filtered_df[selected_numeric].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_numeric}")
    ax.set_xlabel(selected_numeric)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=filtered_df, x="churn", y=selected_numeric, ax=ax)
    ax.set_title(f"{selected_numeric} by Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(selected_numeric)
    st.pyplot(fig)

with tab4:
    st.header("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between Numeric Columns")
    st.pyplot(fig)

with tab5:
    st.header("Custom Plot")
    numeric_columns = filtered_df.select_dtypes(include=np.number).columns.tolist()
    x_col = st.selectbox("Select X-axis", numeric_columns, index=0)
    y_col = st.selectbox("Select Y-axis", numeric_columns, index=min(1, len(numeric_columns)-1))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, hue="churn", ax=ax)
    ax.set_title(f"{x_col} vs {y_col}")
    st.pyplot(fig)

st.subheader("Simple Insights")
if len(filtered_df) > 0:
    most_common_plan = filtered_df["plan_type"].mode()[0]
    avg_usage = filtered_df["avg_weekly_usage_hours"].mean()
    avg_tickets = filtered_df["support_tickets"].mean()
    st.write(f"- Most common plan type: **{most_common_plan}**")
    st.write(f"- Average weekly usage: **{avg_usage:.2f} hours**")
    st.write(f"- Average support tickets: **{avg_tickets:.2f}**")
    st.write(f"- Churn rate in selected data: **{churn_rate:.2f}%**")
else:
    st.warning("No records available after applying filters.")
