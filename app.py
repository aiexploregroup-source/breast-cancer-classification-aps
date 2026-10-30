import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# 🔹 Replace with your actual Google Drive File ID
FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"  # <-- Update with your actual ID
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_PATH = "model.h5"

# 🔹 Function to download and load the model
@st.cache_resource
def load_model():
    if not os.path.exists(OUTPUT_PATH):  # Check if model already exists
        st.write("📥 Downloading model...")
        try:
            gdown.download(URL, OUTPUT_PATH, quiet=False)
            st.write("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None

    if os.path.exists(OUTPUT_PATH):
        model = tf.keras.models.load_model(OUTPUT_PATH)
        st.write(f"✅ Model loaded! Expected input shape: {model.input_shape}")
        return model
    else:
        st.error("❌ Model file not found after download.")
        return None

# 🔹 Load the trained model
model = load_model()
if model is None:
    st.stop()

# 🔹 Define class labels
CLASS_NAMES = ["Benign", "Malignant"]

# 🔹 Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 color channels
    image = image.resize((48, 48))  # Resize to match model input (48x48 pixels)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values (0-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 3)
    
    return image

# 🔹 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.write("Use the menu below to navigate.")
page = st.sidebar.radio("Select a Page:", ["🏠 Home", "📤 Upload & Predict", "ℹ️ About"])

if page == "🏠 Home":
    st.title("🏠 Breast Cancer Classifier")
    st.write("""
    Welcome! This app classifies breast cancer cell images as **Benign** or **Malignant**.
    
    🔹 **How It Works:**  
    - Upload an image of a breast cancer cell  
    - The model analyzes the image  
    - It predicts if the cell is **Benign** or **Malignant**
    
    👉 Click on "Upload & Predict" in the sidebar to start!
    """)

elif page == "📤 Upload & Predict":
    st.title("📤 Upload & Predict")

    # 🔹 File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

        with col2:
            st.write("✅ **Image uploaded successfully!**")
            st.write(f"📏 **Image Size:** {image.size}")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Debugging: Print shape before prediction
        st.write(f"📌 **Processed Image Shape:** {processed_image.shape}")

        # Prediction with progress bar
        with st.spinner("🧐 Analyzing image... Please wait..."):
            progress_bar = st.progress(0)
            for percent in range(100):
                progress_bar.progress(percent + 1)
            
            prediction = model.predict(processed_image)
            class_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        # Display results with emojis
        result = "🟢 **Benign** 😊" if class_index == 0 else "🔴 **Malignant** 😔"
        st.subheader(f"📌 Prediction: {result}")
        st.write(f"✅ **Confidence: {confidence:.2f}%**")

        # Display additional message based on prediction
        if class_index == 0:  # Benign
            st.success("🟢 No cancer detected. Live happy! 😊")
        else:  # Malignant
            st.error("🔴 Cancer detected. Please reach out to a doctor immediately. 🚨")

elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.write("""
    **Breast Cancer Classification App**  
    - This application uses a **Convolutional Neural Network (CNN)** to classify breast cancer cell images.  
    - It was trained using a **Custom CNN model** with TensorFlow & Keras.  
    - The app is deployed using **Streamlit Cloud**.  
      
    **👩‍💻 Developed by:** Team Breast Scan AI from APS LBS Marg , Lucknow 
    **📅 Year:** 2026  
    **🔗 GitHub:** [Your GitHub Repo](https://github.com/SwaroopaNekkanti)  
    """)

# 🔹 Custom Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #1c1c1e;
        text-align: center;
        padding: 10px;
        color: white;
    }
    </style>
    <div class="footer">
        <p>© 2026 Breast Cancer Classifier | Developed by Team Breast Scan AI </p>
    </div>
    """,
    unsafe_allow_html=True
)
