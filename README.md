# 🚗 Smart Car Price Prediction App

A complete machine learning pipeline that predicts the market price of used cars based on real-world attributes. The project includes data cleaning, feature engineering, model training, SHAP explainability, and a deployed **Streamlit** web application.

---

## 🔍 Project Overview
This app enables users to:
- Analyze a real-world car pricing dataset with over 19,000 entries
- Explore data insights (EDA)
- Input car specifications to predict price
- View prediction explanations using SHAP
- Use a clean, navigable interface with **Streamlit**

Model used: **Random Forest Regressor**  
Accuracy: **R² ≈ 0.82**

---

## 📂 Repository Structure
```
car-price-predictor/
│
├── app/                         # Application and config files
│   └── full_car_price_app.py    # Streamlit web app
│
├── models/                      # Saved model and preprocessor
│   ├── best_model.pkl
│   └── preprocessor.pkl
│
├── data/
│   └── car_price_prediction.csv # Cleaned dataset
│
├── assets/                      # Images, banners, logos
│   └── banner.png
│
├── notebooks/                   # EDA and training notebooks
│   └── model_training.ipynb
│
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

---

## 📊 Dataset
- 18 columns, 19,000+ rows
- Target: `Price`
- Features: `Manufacturer`, `Engine volume`, `Mileage`, `Fuel type`, `Gearbox`, `Drive wheels`, `Levy`, `Airbags`, etc.

---

## 🚀 How to Run the App

### 🔧 1. Clone the repository
```bash
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

### 📦 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 📥 3. Install dependencies
```bash
pip install -r requirements.txt
```

### ▶️ 4. Run the Streamlit app
```bash
streamlit run app/full_car_price_app.py
```

---

## 🌐 Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and select this repo
4. Set `app/full_car_price_app.py` as the entry point

Streamlit will automatically install from `requirements.txt` and launch your app!

---

## 📌 Highlights
- ML Pipeline: Clean → Feature Engineer → Train → Predict → Explain
- Model: Random Forest with Pickle I/O
- EDA: Price distribution, correlations, top brands
- Navigation: Sidebar-driven layout with 3 sections

---

## 📘 License
This project is licensed under the MIT License.

---

## ✨ Acknowledgments
- [Streamlit](https://streamlit.io/)
- Dataset adapted for educational purposes

> Built with ❤️ by [Abdelmoneim Moustafa]
