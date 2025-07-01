import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

# -------------------- Page Configuration -------------------- #
st.set_page_config(page_title="Car Price Predictor", page_icon="🚘", layout="wide")

# -------------------- Load Data & Model -------------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("car_price_prediction.csv")
    df['Mileage'] = df['Mileage'].str.replace(' km', '').str.replace(',', '').astype(int)
    df['Engine volume'] = df['Engine volume'].str.replace(' Turbo', '').astype(float)
    df['Age'] = datetime.now().year - df['Prod. year']
    return df

data = load_data()

@st.cache_resource
def load_model():
    try:
        with open("Car_Prediction.sav", 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model = load_model()

# -------------------- Encoders -------------------- #
Manufacturer_mapping = {m: i for i, m in enumerate(sorted(data['Manufacturer'].unique()))}
Model_mapping = {m: i for i, m in enumerate(sorted(data['Model'].unique()))}
Category_mapping = {m: i for i, m in enumerate(sorted(data['Category'].unique()))}
leather_mapping = {'yes': 1, 'no': 2}
Fuel_mapping = {m: i for i, m in enumerate(sorted(data['Fuel type'].unique()))}
Gear_mapping = {m: i for i, m in enumerate(sorted(data['Gear box type'].unique()))}
Drive_mapping = {m: i for i, m in enumerate(sorted(data['Drive wheels'].unique()))}
Wheel_mapping = {m: i for i, m in enumerate(sorted(data['Wheel'].unique()))}
color_mapping = {m: i for i, m in enumerate(sorted(data['Color'].unique()))}

# -------------------- Sidebar Navigation -------------------- #
with st.sidebar:
    st.title("📌 Navigation")
    nav = st.radio("Select a Section:", ["📖 Overview", "📊 EDA", "🧮 Predict"])
    st.markdown("---")
    st.markdown("[🔗 GitHub Repository](https://github.com/Abdelmoneim-Moustafa/Car_price_prediction)")

# -------------------- Banner -------------------- #
st.markdown("""
<div style='height:300px;background-image: url("https://wallpapercave.com/wp/wp3202836.jpg");
     background-size: cover; background-position: center; border-radius: 12px;'>
</div>
""", unsafe_allow_html=True)

# -------------------- Overview -------------------- #
if nav == "📖 Overview":
    st.title("🚗 Smart Car Price Predictor")
    st.markdown("""
    Welcome to the **Smart Car Price Predictor** — a powerful tool to analyze car pricing trends and estimate fair market values.

    ### 🚀 Key Features:
    - Market EDA visualizations
    - ML-powered price prediction
    - Interactive interface

    ### 🛠 Built with: Streamlit · Pandas · Plotly · Scikit-learn
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Listings", f"{len(data):,}")
    col2.metric("Avg. Price", f"${int(data['Price'].mean()):,}")
    col3.metric("Top Brand", data['Manufacturer'].value_counts().idxmax())

    st.markdown("### 🔍 Sample Data")
    st.dataframe(data.head(), use_container_width=True)

# -------------------- EDA -------------------- #
elif nav == "📊 EDA":
    st.title("📊 EDA Dashboard")
    st.markdown("### 🎯 Filter Options")

    price_ranges = {
        "1K to 5K": (1000, 5000),
        "6K to 10K": (6000, 10000),
        "11K to 20K": (11000, 20000),
        "21K to 50K": (21000, 50000),
        "Above 50K": (50001, int(data['Price'].max()))
    }
    selected_range = st.selectbox("Price Range:", list(price_ranges.keys()))
    price_min, price_max = price_ranges[selected_range]

    selected_brands = st.multiselect("Select Brands:", data['Manufacturer'].unique(),
                                     default=data['Manufacturer'].value_counts().nlargest(3).index.tolist())

    df_filtered = data[(data['Price'] >= price_min) & (data['Price'] <= price_max)]
    if selected_brands:
        df_filtered = df_filtered[df_filtered['Manufacturer'].isin(selected_brands)]

    tab1, tab2, tab3, tab4 = st.tabs(["📆 Price by Year", "🏆 Top Brands", "📊 Distributions", "📌 Correlations"])

    with tab1:
        yearly = df_filtered.groupby('Prod. year')['Price'].mean().reset_index()
        st.plotly_chart(px.line(yearly, x='Prod. year', y='Price', markers=True), use_container_width=True)

    with tab2:
        top_counts = df_filtered['Manufacturer'].value_counts().nlargest(10).reset_index()
        top_counts.columns = ['Manufacturer', 'Count']
        st.plotly_chart(px.bar(top_counts, x='Manufacturer', y='Count', color='Manufacturer'), use_container_width=True)

        avg_price = df_filtered.groupby('Manufacturer')['Price'].mean().loc[top_counts['Manufacturer']].reset_index()
        avg_price.columns = ['Manufacturer', 'AvgPrice']
        st.plotly_chart(px.line(avg_price, x='Manufacturer', y='AvgPrice', markers=True), use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        col1.plotly_chart(px.histogram(df_filtered, x='Price', nbins=50), use_container_width=True)
        col1.plotly_chart(px.box(df_filtered, x='Fuel type', y='Price', color='Fuel type'), use_container_width=True)
        col2.plotly_chart(px.histogram(df_filtered, x='Mileage', nbins=50), use_container_width=True)
        col2.plotly_chart(px.box(df_filtered, x='Category', y='Price', color='Category'), use_container_width=True)

    with tab4:
        corr = df_filtered[['Price', 'Mileage', 'Engine volume', 'Airbags', 'Cylinders', 'Age']].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", width=1200, height=800)
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------- Prediction -------------------- #
elif nav == "🧮 Predict":
    st.title("🧮 Car Price Estimator")

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Manufacturer", list(Manufacturer_mapping.keys()))
        Manufacturer = Manufacturer_mapping[brand]

        available_models = sorted(data[data['Manufacturer'] == brand]['Model'].unique())
        model_label = st.selectbox("Model", available_models)
        Model = Model_mapping[model_label]  # USE GLOBAL MAPPING HERE

        Category = Category_mapping[st.selectbox("Category", list(Category_mapping.keys()))]
        Leather = leather_mapping[st.selectbox("Leather Interior", list(leather_mapping.keys()))]
        Fuel = Fuel_mapping[st.selectbox("Fuel Type", list(Fuel_mapping.keys()))]
        Mileage = st.number_input("Mileage (km)", min_value=0)

    with col2:
        Gear = Gear_mapping[st.selectbox("Gearbox Type", list(Gear_mapping.keys()))]
        Drive = Drive_mapping[st.selectbox("Drive Wheels", list(Drive_mapping.keys()))]
        Wheel = Wheel_mapping[st.selectbox("Wheel Position", list(Wheel_mapping.keys()))]
        Color = color_mapping[st.selectbox("Color", list(color_mapping.keys()))]
        Engine = st.selectbox("Engine Volume", sorted(data['Engine volume'].unique()))
        Airbags = st.selectbox("Airbags", sorted(data['Airbags'].dropna().unique()))
        Age = st.number_input("Vehicle Age", min_value=0)
        Levy = st.number_input("Levy Tax", min_value=0)

    if st.button("🔍 Estimate Price"):
        input_data = pd.DataFrame({
            'Manufacturer': [Manufacturer], 'Model': [Model], 'Category': [Category],
            'Leather interior': [Leather], 'Fuel type': [Fuel], 'Mileage': [Mileage],
            'Gear box type': [Gear], 'Drive wheels': [Drive], 'Wheel': [Wheel],
            'Color': [Color], 'Levy': [Levy], 'Engine volume': [Engine],
            'Airbags': [Airbags], 'Age': [Age]
        })

        try:
            prediction = model.predict(input_data)[0]
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 12px; background: #f0f9f9; border-left: 6px solid #00cc66;">
                <h3 style="color:#007a5a;">💰 Estimated Price:</h3>
                <h1 style="color:#007a5a;">${prediction:,.2f}</h1>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 🔎 Input Summary")
            st.dataframe(input_data, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -------------------- Footer -------------------- #
st.markdown("---")
st.markdown("### Project on GitHub")
st.markdown("[https://github.com/Abdelmoneim-Moustafa/Car_price_prediction](https://github.com/Abdelmoneim-Moustafa/Car_price_prediction)")
st.caption("Car Price Predictor · Author: Abdelmoneim Behairy")
