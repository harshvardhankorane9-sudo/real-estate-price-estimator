# 🏠 Bengaluru House Price Prediction

This is a simple machine learning project where I built an interactive web app that predicts house prices in Bengaluru based on parameters like area type, location, BHK, bathrooms, and total square feet.

I wanted to make something that’s not just about model training, but also looks good and is easy to use — so I built a small Streamlit app that gives instant predictions.

---

## 🔍 About the Project

The model is trained on a real Bengaluru housing dataset.  
I cleaned and processed the data using Pandas and NumPy, removed outliers, handled rare locations, and then used **Linear Regression** for prediction.

The app can take simple inputs from the user and shows the estimated price in **Lakhs** or **Crores** automatically.  
For example, if the prediction is 193 Lakhs, it shows it as **1.93 Crore**.

> **Note:** Due to size constraints, the full dataset is **not included** in this repo.  
> You can download the complete dataset from Kaggle here:  
> 👉 [Bengaluru House Dataset](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

---

## ⚙️ Tech Stack

- Python  
- Pandas, NumPy, Scikit‑Learn  
- Matplotlib, Seaborn  
- Streamlit for UI  