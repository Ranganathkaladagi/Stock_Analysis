import streamlit as st
import pandas as pd

from data import get_data
from preprocess import preprocess
from model import train_model, predict

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Stock Market Analysis and Prediction",
    layout="wide"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: white;
}

h1, h2, h3, h4, h5 {
    color: black;
}

p, label, div {
    color: black;
}

.stButton > button {
    background-color: #007BFF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.stDownloadButton > button {
    background-color: green;
    color: white;
    border-radius: 10px;
}

section[data-testid="stSidebar"] {
    background-color: #F0F2F6;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("📈 Stock Market Analysis and Prediction")

st.write(
    "Machine Learning Based Stock Market Analysis & Prediction System"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("User Input")

company_names = {
    "Apple Inc. (AAPL)": "AAPL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "Infosys (INFY.NS)": "INFY.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Reliance Industries (RELIANCE.NS)": "RELIANCE.NS",
    "Other": "OTHER"
}

selected_company = st.sidebar.selectbox(
    "Select Stock Symbol",
    list(company_names.keys())
)

# =====================================================
# OTHER STOCK OPTION
# =====================================================

if selected_company == "Other":

    stock = st.sidebar.text_input(
        "Enter Stock Symbol",
        placeholder="Example: IBM, NVDA, SBI.NS"
    )

    company_display = stock

else:

    stock = company_names[selected_company]
    company_display = selected_company

# =====================================================
# DATE INPUTS
# =====================================================

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2020-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

# =====================================================
# SUBMIT BUTTON
# =====================================================

submit = st.sidebar.button("🚀 Submit")

# =====================================================
# MAIN APPLICATION
# =====================================================

if submit:

    if stock == "":
        st.warning("⚠️ Please enter a stock symbol.")
        st.stop()

    with st.spinner("Fetching stock market data..."):

        data = get_data(
            stock,
            start_date,
            end_date
        )

    if data.empty:

        st.error("❌ No stock data found.")
        st.stop()

    # Preprocess
    data = preprocess(data)

    # Train model
    model, accuracy = train_model(data)

    # Current Price
    current_price = data["Close"].iloc[-1]

    # Highest & Lowest Price
    highest_price = data["High"].max()
    lowest_price = data["Low"].min()

    # Currency
    if ".NS" in stock:
        currency = "₹"
    else:
        currency = "$"

    # =====================================================
    # COMPANY NAME
    # =====================================================

    st.header(company_display)

    # =====================================================
    # OVERVIEW METRICS
    # =====================================================

    st.subheader("📊 Stock Market Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Price",
            f"{currency}{current_price:.2f}"
        )

    with col2:
        st.metric(
            "Highest Price",
            f"{currency}{highest_price:.2f}"
        )

    with col3:
        st.metric(
            "Lowest Price",
            f"{currency}{lowest_price:.2f}"
        )

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs([
        "📈 Charts",
        "🤖 Prediction",
        "📋 Dataset"
    ])

    # =====================================================
    # CHARTS TAB
    # =====================================================

    with tab1:

        st.subheader("📈 Closing Price Chart")

        st.line_chart(
            data["Close"],
            height=400
        )

        st.subheader("📊 Open, High, Low, Close Comparison")

        st.line_chart(
            data[["Open", "High", "Low", "Close"]],
            height=450
        )

        st.subheader("📉 Moving Average Analysis")

        st.line_chart(
            data[["Close", "MA20", "MA50"]],
            height=450
        )

        st.subheader("📈 Stock Trend Area Chart")

        st.area_chart(
            data["Close"]
        )

        st.subheader("📊 Trading Volume")

        st.bar_chart(
            data["Volume"]
        )

    # =====================================================
    # PREDICTION TAB
    # =====================================================

    with tab2:

        st.subheader("🤖 Model Accuracy")

        st.success(
            f"Accuracy: {accuracy:.2f}"
        )

        st.progress(float(accuracy))

        latest_ma20 = data["MA20"].iloc[-1]
        latest_ma50 = data["MA50"].iloc[-1]

        result = predict(
            model,
            latest_ma20,
            latest_ma50
        )

        st.subheader("🔮 Tomorrow Prediction")

        if result[0] == 1:

            st.success(
                "📈 Stock Price May Go UP Tomorrow"
            )

            st.success(
                "✅ Recommendation: BUY"
            )


        else:

            st.error(
                "📉 Stock Price May Go DOWN Tomorrow"
            )

            st.error(
                "❌ Recommendation: SELL"
            )

    # =====================================================
    # DATASET TAB
    # =====================================================

    with tab3:

        st.subheader("📋 Recent Stock Data")

        st.dataframe(
            data.tail(10),
            use_container_width=True
        )

        with st.expander("📄 View Full Dataset"):

            st.dataframe(
                data,
                use_container_width=True
            )

        csv = data.to_csv().encode("utf-8")

        st.download_button(
            label="📥 Download Dataset",
            data=csv,
            file_name=f"{stock}_data.csv",
            mime="text/csv"
        )

else:

    st.info(
        "👈 Select a company and click Submit from the sidebar."
    )