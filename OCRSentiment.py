import streamlit as st
import pytesseract
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="OCR and Sentiment Analysis App", page_icon=":rocket:")
st.header("Optical Character Recognition and Sentiment Analysis")

st.write(
    """
    <style>
    .stApp {
        background-color: #f0f8ff; /* Light Blue */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["OCR", "Sentiment Analysis"])
st.write("Select the app mode from the sidebar to use OCR or Sentiment Analysis.")

if app_mode == "OCR":
    st.title("OCR - Scan Text in Image 📜")
    st.markdown("---")

    uploaded_image = st.file_uploader("Select an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("---")

        if st.button("Click to Perform OCR"):
            extracted_text = pytesseract.image_to_string(image)

            st.header("Extracted Text:")
            st.write(extracted_text)

elif app_mode == "Sentiment Analysis":
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter a message:")

    if st.button("Analyze Sentiment"):
        if user_input:
            # Load the sentiment analysis model
            model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"
            sentiment_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

            # Perform sentiment analysis
            results = sentiment_classifier(user_input)
            sentiment_label = results[0]["label"]
            sentiment_score = results[0]["score"]

            st.write(f"Sentiment: {sentiment_label}")
            st.write(f"Confidence Score: {sentiment_score:.2f}")
