""import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

\# 🔹 Replace with your actual Google Drive File ID
FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"  # <-- Update with your actual ID
URL = f"[https://drive.google.com/uc?export=download&id={FILE_ID](https://drive.google.com/uc?export=download\&id={FILE_ID)}"
OUTPUT_PATH = "model.h5"

\# 🔹 Function to download and load the model
@st.cache_resource
def load_model():
&#x20;   if not os.path.exists(OUTPUT_PATH):  # Check if model already exists
&#x20;       st.write("📥 Downloading model...")
&#x20;       try:
&#x20;           gdown.download(URL, OUTPUT_PATH, quiet=False)
&#x20;           st.write("✅ Model downloaded successfully!")
&#x20;       except Exception as e:
&#x20;           st.error(f"❌ Model download failed: {e}")
&#x20;           return None

&#x20;   if os.path.exists(OUTPUT_PATH):
&#x20;       model = tf.keras.models.load_model(OUTPUT_PATH)
&#x20;       st.write(f"✅ Model loaded! Expected input shape: {model.input_shape}")
&#x20;       return model
&#x20;   else:
&#x20;       st.error("❌ Model file not found after download.")
&#x20;       return None

\# 🔹 Load the trained model
model = load_model()
if model is None:
&#x20;   st.stop()

\# 🔹 Define class labels
CLASS_NAMES = ["Benign", "Malignant"]

\# 🔹 Function to preprocess image
def preprocess_image(image):
&#x20;   image = image.convert("RGB")  # Ensure 3 color channels
&#x20;   image = image.resize((48, 48))  # Resize to match model input (48x48 pixels)
&#x20;   image = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values (0-1)
&#x20;   image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 48, 48, 3)
&#x20;  &#x20;
&#x20;   return image

\# 🔹 Sidebar Navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.write("Use the menu below to navigate.")
page = st.sidebar.radio("Select a Page:", ["🏠 Home", "📤 Upload & Predict", "ℹ️ About"])

if page == "🏠 Home":
&#x20;   st.title("🏠 Breast Cancer Classifier")
&#x20;   st.write("""
&#x20;   Welcome! This app classifies breast cancer cell images as \*\*Benign\*\* or \*\*Malignant\*\*.
&#x20;  &#x20;
&#x20;   🔹 \*\*How It Works:\*\* &#x20;
&#x20;   \- Upload an image of a breast cancer cell &#x20;
&#x20;   \- The model analyzes the image &#x20;
&#x20;   \- It predicts if the cell is \*\*Benign\*\* or \*\*Malignant\*\*
&#x20;  &#x20;
&#x20;   👉 Click on "Upload & Predict" in the sidebar to start!
&#x20;   """)

elif page == "📤 Upload & Predict":
&#x20;   st.title("📤 Upload & Predict")

&#x20;   \# 🔹 File uploader
&#x20;   uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

&#x20;   if uploaded_file:
&#x20;       col1, col2 = st.columns(2)
&#x20;      &#x20;
&#x20;       with col1:
&#x20;           image = Image.open(uploaded_file)
&#x20;           st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

&#x20;       with col2:
&#x20;           st.write("✅ \*\*Image uploaded successfully!\*\*")
&#x20;           st.write(f"📏 \*\*Image Size:\*\* {image.size}")

&#x20;       \# Preprocess the image
&#x20;       processed_image = preprocess_image(image)

&#x20;       \# Debugging: Print shape before prediction
&#x20;       st.write(f"📌 \*\*Processed Image Shape:\*\* {processed_image.shape}")

&#x20;       \# Prediction with progress bar
&#x20;       with st.spinner("🧐 Analyzing image... Please wait..."):
&#x20;           progress_bar = st.progress(0)
&#x20;           for percent in range(100):
&#x20;               progress_bar.progress(percent + 1)
&#x20;          &#x20;
&#x20;           prediction = model.predict(processed_image)
&#x20;           class_index = np.argmax(prediction)
&#x20;           confidence = np.max(prediction) \* 100

&#x20;       \# Display results with emojis
&#x20;       result = "🟢 \*\*Benign\*\* 😊" if class_index == 0 else "🔴 \*\*Malignant\*\* 😔"
&#x20;       st.subheader(f"📌 Prediction: {result}")
&#x20;       st.write(f"✅ \*\*Confidence: {confidence:.2f}%\*\*")

&#x20;       \# Display additional message based on prediction
&#x20;       if class_index == 0:  # Benign
&#x20;           st.success("🟢 No cancer detected. Live happy! 😊")
&#x20;       else:  # Malignant
&#x20;           st.error("🔴 Cancer detected. Please reach out to a doctor immediately. 🚨")

elif page == "ℹ️ About":
&#x20;   st.title("ℹ️ About This Project")
&#x20;   st.write("""
&#x20;   \*\*Breast Cancer Classification App\*\* &#x20;
&#x20;   \- This application uses a \*\*Convolutional Neural Network (CNN)\*\* to classify breast cancer cell images. &#x20;
&#x20;   \- It was trained using a \*\*Custom CNN model\*\* with TensorFlow & Keras. &#x20;
&#x20;   \- The app is deployed using \*\*Streamlit Cloud\*\*. &#x20;
&#x20;    &#x20;
&#x20;   \*\*👩‍💻 Developed by:\*\* Team Breast Scan AI from APS LBS Marg , Lucknow&#x20;
&#x20;   \*\*📅 Year:\*\* 2026 &#x20;
&#x20;   \*\*🔗 GitHub:\*\* [https://github.com/aiexploregroup-source/breast-cancer-classification-aps](https://github.com/aiexploregroup-source/breast-cancer-classification-aps) &#x20;
&#x20;   """)

\# 🔹 Custom Footer
st.markdown(
&#x20;   """
&#x20;   \<style>
&#x20;   .footer {
&#x20;       position: fixed;
&#x20;       bottom: 0;
&#x20;       left: 0;
&#x20;       width: 100%;
&#x20;       background-color: #1c1c1e;
&#x20;       text-align: center;
&#x20;       padding: 10px;
&#x20;       color: white;
&#x20;   }
&#x20;   \</style>
&#x20;   \<div class="footer">
&#x20;       \<p>© 2026 Breast Cancer Classifier | Developed by Team Breast Scan AI \</p>
&#x20;   \</div>
&#x20;   """,
&#x20;   unsafe_allow_html=True
)"" enhancet the complete design of the website but dint try t use exteral libraries
