import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Styling

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


# Load & Train Model

@st.cache_data
def load_model():
    data = pd.read_csv("data/train.csv")

    data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath',
                 'OverallQual', 'GarageCars', 'SalePrice']]

    data = data.dropna()

    X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath',
              'OverallQual', 'GarageCars']]
    y = data['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

model = load_model()


#UI

st.title("🏠 House Price Predictor")
st.write("Enter house details to predict price")

# Inputs
area = st.number_input("Square Footage (GrLivArea)", 500, 5000, 2000)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 5, 2)
quality = st.slider("Overall Quality (1-10)", 1, 10, 5)
garage = st.number_input("Garage Capacity", 0, 5, 2)


# Prediction

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, quality, garage]],
                             columns=['GrLivArea', 'BedroomAbvGr', 'FullBath',
                                      'OverallQual', 'GarageCars'])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")

    # Smart message
    if prediction[0] > 300000:
        st.info("🏡 This is a high-value property!")
    else:
        st.info("🏠 This is a moderately priced house.")