import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import time
import random

st.set_page_config(page_title="AI Commerce Intelligence", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.big-font {
    font-size:22px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD (CACHED) ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/gold/data.csv")
    return df.sample(min(len(df), 50000))

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# ---------------- LOADING EFFECT ----------------
placeholder = st.empty()
placeholder.text("🔄 Updating live data...")
time.sleep(1)
placeholder.empty()

# 🔥 FIXED-AREA SLIDESHOW (ONLY IMAGE CHANGES VISUALLY)
st.title("🚀 AI Powered E-Commerce Intelligence Dashboard")

# ---------------- HERO IMAGE ----------------
st.image(
    "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
    use_container_width=True
)

# ---------------- HERO IMAGE ----------------


# ---------------- SIDEBAR BRANDING ----------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=100
)
st.sidebar.title("AI Commerce App")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

users = st.sidebar.multiselect("Users", df["user_id"].unique())
products = st.sidebar.multiselect("Products", df["product_id"].unique())
countries = st.sidebar.multiselect("Countries", df["country"].unique())

min_q, max_q = st.sidebar.slider(
    "Quantity Range",
    int(df["quantity"].min()),
    int(df["quantity"].max()),
    (int(df["quantity"].min()), int(df["quantity"].max()))
)

filtered_df = df

if users:
    filtered_df = filtered_df[filtered_df["user_id"].isin(users)]
if products:
    filtered_df = filtered_df[filtered_df["product_id"].isin(products)]
if countries:
    filtered_df = filtered_df[filtered_df["country"].isin(countries)]

filtered_df = filtered_df[
    (filtered_df["quantity"] >= min_q) &
    (filtered_df["quantity"] <= max_q)
]

# ---------------- KPI ----------------
st.subheader("📊 Business Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Revenue", f"₹ {filtered_df['total'].sum():,.0f}")
col2.metric("📦 Orders", len(filtered_df))
col3.metric("📊 Avg Order", f"₹ {filtered_df['total'].mean():.0f}")

top_country = "N/A"
if len(filtered_df) > 0:
    top_country = filtered_df.groupby("country")["total"].sum().idxmax()

col4.metric("🌍 Top Country", top_country)

# ---------------- LIVE KPI ----------------
live_revenue = filtered_df["total"].sum() + random.randint(0, 500)
st.metric("⚡ Live Revenue", f"₹ {live_revenue}")

st.divider()

# ---------------- DOWNLOAD ----------------
st.download_button(
    "📥 Download Filtered Data",
    filtered_df.to_csv(index=False),
    file_name="filtered_data.csv"
)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "🤖 Prediction",
    "📈 Insights",
    "📉 Simulator",
    "🧠 Advanced",
    "🚀 AI Lab",
    "💬 Chat"
])

# ================= DASHBOARD =================
with tab1:
    st.subheader("📊 Analytics")

    col1, col2 = st.columns(2)

    with col1:
        user_df = filtered_df.groupby("user_id")["total"].sum().nlargest(20).reset_index()
        fig = px.bar(user_df, x="user_id", y="total", title="Top Users by Revenue")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prod_df = filtered_df.groupby("product_id")["total"].sum().nlargest(20).reset_index()
        fig = px.bar(prod_df, x="product_id", y="total", title="Top Products")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌍 Country Distribution")

    country_df = filtered_df.groupby("country")["total"].sum().reset_index()

    fig = px.pie(country_df, names="country", values="total", title="Revenue Share")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌍 Global Revenue Map")

    if len(country_df) > 0:
        fig = px.choropleth(
            country_df,
            locations="country",
            locationmode="country names",
            color="total",
            title="Global Revenue Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No country data available")

# ================= PREDICTION =================
with tab2:
    st.subheader("🤖 AI Prediction")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Price", min_value=0.0, value=100.0)

    with col2:
        quantity = st.slider("Quantity", 1, 10, 1)

    avg_price = price

    if st.button("Predict Revenue"):
        pred = model.predict([[price, quantity, avg_price]])
        st.success(f"💰 Predicted Revenue: ₹ {round(pred[0], 2)}")

    st.subheader("📂 Bulk Prediction")

    uploaded_file = st.file_uploader("Upload CSV with price column")

    if uploaded_file:
        temp_df = pd.read_csv(uploaded_file)

        if "price" in temp_df.columns:
            temp_df["quantity"] = 1
            temp_df["avg_price_per_user"] = temp_df["price"]

            temp_df["predicted_revenue"] = model.predict(
                temp_df[["price", "quantity", "avg_price_per_user"]]
            )

            st.dataframe(temp_df)
        else:
            st.error("CSV must contain 'price' column")

# ================= INSIGHTS =================
with tab3:
    st.subheader("📈 Smart Insights")

    if len(filtered_df) > 0:
        top_user = filtered_df.groupby("user_id")["total"].sum().idxmax()
        top_product = filtered_df.groupby("product_id")["total"].sum().idxmax()

        st.success(f"🔥 Top User: {top_user}")
        st.success(f"🔥 Top Product: {top_product}")
    else:
        st.warning("No data after filters")

    st.subheader("📦 Quantity Distribution")
    fig = px.histogram(filtered_df, x="quantity")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Revenue Trend")
    fig = px.line(filtered_df.head(1000), y="total")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Correlation Heatmap")
    corr = filtered_df[["price", "quantity", "total"]].corr()
    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

# ================= SIMULATOR =================
with tab4:
    st.subheader("📉 What-if Simulator")

    price_range = st.slider("Price Range", 10, 1000, (50, 500))

    prices = np.arange(price_range[0], price_range[1], 20)
    preds = model.predict([[p, 2, p] for p in prices])

    sim_df = pd.DataFrame({
        "Price": prices,
        "Revenue": preds
    })

    fig = px.line(sim_df, x="Price", y="Revenue", title="Revenue Simulation")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(sim_df.head(20))

# ================= ADVANCED =================
with tab5:
    st.subheader("🧠 Advanced Analytics")

    if len(filtered_df) == 0:
        st.warning("No data after filters.")
    else:
        rfm = filtered_df.groupby("user_id").agg({
            "total": ["sum", "mean"],
            "quantity": "sum"
        })

        rfm.columns = ["total_spent", "avg_spent", "total_qty"]

        rfm["segment"] = np.where(
            rfm["total_spent"] > rfm["total_spent"].quantile(0.75),
            "High Value",
            np.where(
                rfm["total_spent"] > rfm["total_spent"].quantile(0.4),
                "Mid Value",
                "Low Value"
            )
        )

        st.dataframe(rfm.head(20))

        fig = px.pie(rfm.reset_index(), names="segment")
        st.plotly_chart(fig, use_container_width=True)

        threshold = filtered_df["total"].mean() + 2 * filtered_df["total"].std()
        temp_df = filtered_df.copy()
        temp_df["anomaly"] = temp_df["total"] > threshold

        fig = px.scatter(temp_df.head(1000), x="price", y="total", color="anomaly")
        st.plotly_chart(fig)

# ================= AI LAB =================
with tab6:
    st.subheader("🚀 AI Decision Lab")

    if len(filtered_df) == 0:
        st.warning("No data available")
    else:
        base_price = st.slider("Base Price", 10, 1000, 100)

        test_prices = np.arange(base_price * 0.5, base_price * 1.5, 10)
        preds = model.predict([[p, 2, p] for p in test_prices])

        opt_df = pd.DataFrame({
            "Price": test_prices,
            "Revenue": preds
        })

        best_price = opt_df.loc[opt_df["Revenue"].idxmax()]

        st.success(f"🔥 Optimal Price: ₹ {round(best_price['Price'], 2)}")

        fig = px.line(opt_df, x="Price", y="Revenue")
        st.plotly_chart(fig)

        clv = filtered_df.groupby("user_id")["total"].sum().reset_index()
        st.dataframe(clv.sort_values("total", ascending=False).head(10))

        repeat_users = filtered_df["user_id"].value_counts()
        repeat_rate = (repeat_users > 1).sum() / len(repeat_users) if len(repeat_users) > 0 else 0

        st.metric("Repeat Customer Rate", f"{round(repeat_rate * 100, 2)} %")

# ================= CHAT =================
with tab7:
    st.subheader("💬 Ask Your Data")

    question = st.text_input("Ask something like: top country?")

    if question:
        if "country" in question.lower():
            st.write(filtered_df.groupby("country")["total"].sum().idxmax())
        elif "revenue" in question.lower():
            st.write(filtered_df["total"].sum())
        elif "user" in question.lower():
            st.write(filtered_df.groupby("user_id")["total"].sum().idxmax())
        else:
            st.write("Try asking about revenue, country, or users")


