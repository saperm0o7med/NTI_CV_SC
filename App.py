import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ==================================================
# Custom CSS
# ==================================================
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

h1{
    color:#2E8B57;
    text-align:center;
}

.stButton>button{
    width:100%;
    background:#2E8B57;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#1f6b45;
    color:white;
}

.block-container{
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# Load Model & Scaler
# ==================================================
@st.cache_resource
def load_assets():

    model = load_model("Load_iris.keras")

    with open("scaler.pkl","rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_assets()

# ==================================================
# Title
# ==================================================
st.title("🌸 Iris Flower Classification")

st.write(
"""
Predict the species of an Iris flower using a trained
Deep Learning model built with TensorFlow/Keras.
"""
)

st.divider()

# ==================================================
# Input Section
# ==================================================
left, right = st.columns(2)

with left:

    sepal_length = st.slider(
        "Sepal Length",
        4.0,8.0,5.8,0.1
    )

    sepal_width = st.slider(
        "Sepal Width",
        2.0,4.5,3.0,0.1
    )

with right:

    petal_length = st.slider(
        "Petal Length",
        1.0,7.0,4.3,0.1
    )

    petal_width = st.slider(
        "Petal Width",
        0.1,2.5,1.3,0.1
    )

st.write("")

# ==================================================
# Prediction
# ==================================================
if st.button("🔍 Predict Species"):

    sample = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled, verbose=0)

    classes = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    index = np.argmax(prediction)

    species = classes[index]

    confidence = prediction[0][index]

    st.success(f"### 🌼 Prediction: {species}")

    st.write(f"### Confidence: {confidence*100:.2f}%")

    st.progress(float(confidence))

    st.divider()

    descriptions = {

        "Setosa":
        "🌱 Setosa flowers are generally small with short petals and are the easiest species to distinguish.",

        "Versicolor":
        "🌿 Versicolor has medium-sized petals and shares characteristics with both other species.",

        "Virginica":
        "🌸 Virginica typically has the largest petals and is considered the most mature Iris species."

    }

    st.info(descriptions[species])

# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.caption("Developed using TensorFlow • Streamlit • Scikit-Learn")