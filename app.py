# --- Task 3: Cat vs. Dog Streamlit App ---
# This is the "app" script.
# It's NOT going to re-train the model. That would take way too long.
# Instead, it's just going to load the 'svm_model.joblib' file I
# already created with my 'task_03.py' trainer script.
# This makes the app super fast!

import streamlit as st
from PIL import Image  # To handle image uploads
import numpy as np
import cv2  # OpenCV for image processing
from skimage.feature import hog  # For HOG feature extraction
import joblib  # For loading my saved model
import os

# --- 1. Load my Trained Model ---

# I'm telling Streamlit to "cache" this model.
# This means it will only load it from the file ONCE,
# not every time I upload an image. Smart!
@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained SVM model from a .joblib file."""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    else:
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()  # Stop the app if the model isn't there

# --- 2. Define My Image Processing Function ---

# I need to process the user's uploaded image in the *exact* same way
# I processed my training images in 'task_03.py'.
# Same size, same grayscale, same HOG settings.

# This was the size I used in my trainer script
IMG_SIZE = (64, 64)

def process_image(image_file):
    """Takes an uploaded image file, processes it, and returns HOG features."""
    
    # 1. Read the image
    # The file_uploader gives me a 'file-like' object,
    # so I need to read its bytes first
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    
    # Use cv2 to decode the image from this byte array
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Convert to Grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Resize to 64x64
    resized_img = cv2.resize(gray_img, IMG_SIZE)
    
    # 4. Calculate HOG features (must use the same settings as my trainer!)
    hog_features = hog(resized_img,
                       orientations=9,
                       pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2),
                       transform_sqrt=True,
                       block_norm='L2-Hys')
    
    return hog_features

# --- 3. Build the Streamlit App ---

# Set the page title
st.title("🐱🐶 Cat vs. Dog Classifier 🐶🐱")
st.write("My app for Prodigy InfoTech Task 3 (SVM & HOG Features)")

# Load the model!
model = load_model('svm_model.joblib')

if model:  # Only run the app if the model loaded successfully
    
    # This is the "drag and drop" file uploader
    uploaded_file = st.file_uploader("Upload an image of a cat or a dog:",
                                     type=["jpg", "jpeg", "png"])
    
    # This is the main logic
    if uploaded_file is not None:
        
        # Show the image I just uploaded.
        st.image(uploaded_file, caption="You uploaded this!", use_container_width=True)
        
        # Show a "loading" spinner while it processes the image
        with st.spinner("Classifying... 🧠"):
            
            # 1. Process the image to get HOG features
            hog_features = process_image(uploaded_file)
            
            # 2. Reshape the features to what the SVM model expects
            # My HOG features are a flat list, but the model expects a 2D array
            # .reshape(1, -1) means "1 row, and figure out the columns yourself"
            features_for_model = hog_features.reshape(1, -1)
            
            # 3. Make the prediction!
            prediction = model.predict(features_for_model)
            
            # 4. Get the confidence score
            # (I trained my model with probability=True so I can do this)
            probabilities = model.predict_proba(features_for_model)
            confidence = np.max(probabilities) * 100
            
            # --- 5. Show the result! ---
            
            # My labels were 0 = cat, 1 = dog
            if prediction[0] == 0:
                st.success(f"It's a CAT! 🐈")
                st.write(f"I'm **{confidence:.2f}%** sure.")
            else:
                st.success(f"It's a DOG! 🐕")
                st.write(f"I'm **{confidence:.2f}%** sure.")

else:
    st.error("Model is not loaded. Cannot run the app.")