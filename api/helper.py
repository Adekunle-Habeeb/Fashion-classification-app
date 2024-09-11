import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import matplotlib.pyplot as plt

# Streamlit app configuration
st.set_page_config(page_title="Fashion Classifier", page_icon="👚", layout="wide")

# Loading the Keras model
# @st.cache_resource
def load_model():
    return tf.keras.models.load_model("/saved_model/1.keras")

MODEL = tf.keras.models.load_model("/saved_model/1.keras")

# Class names 
CLASS_NAMES = ['Boys-Apparel', 'Boys-Footwear', 'Girls-Apparel', 'Girls-Footwear']

# Streamlit app title and description
st.title("🎨 Fashion Classification App")
st.markdown("""
This app uses a deep learning model to classify clothing images into different categories.
Upload an image to see how it's classified!
""")

# Sidebar with additional information
st.sidebar.header("About")
st.sidebar.info("This app classifies clothing images into the following categories:")
for name in CLASS_NAMES:
    st.sidebar.write(f"- {name}")
st.sidebar.info("Upload a clear image of a single clothing item or footwear for best results.")

# Function to preprocess the image
def preprocess_image(image) -> np.ndarray:
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    return img_array

# Main content
col1, col2 = st.columns(2)

with col1:
    # Restrict image types to only jpg, jpeg, png
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Process the image
        image = preprocess_image(uploaded_file)
        
        # Adding a button to trigger prediction
        if st.button('Classify Image'):
            with st.spinner('Analyzing the image...'):
                predictions = MODEL.predict(image)
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = np.max(predictions[0])

                # Show the result
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}")

                # Display confidence levels for all classes
                fig, ax = plt.subplots()
                y_pos = np.arange(len(CLASS_NAMES))
                ax.barh(y_pos, predictions[0], align='center')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(CLASS_NAMES)
                ax.invert_yaxis()
                ax.set_xlabel('Confidence')
                ax.set_title('Prediction Confidence for Each Category')

                st.pyplot(fig)

with col2:
    st.header("How it works")
    st.write("""
    1. Upload an image of clothing or footwear.
    2. Click the 'Classify Image' button.
    3. The app will analyze the image and predict its category.
    4. Results show the predicted class and confidence levels.
    
    The model uses deep learning to recognize patterns in the image and classify it into one of the predefined categories.
    """)

    st.header("Tips for best results")
    st.write("""
    - Use clear, well-lit images
    - Ensure the item of clothing or footwear is the main focus of the image
    - Avoid cluttered backgrounds
    - Try different angles if the initial classification seems incorrect
    """)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ by Habeeb")