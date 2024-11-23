import os
import streamlit as st
from PIL import Image
import pytesseract
from ultralytics import YOLO
import inflect
import cv2
import pyttsx3
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure Tesseract executable path
if os.name == 'nt':  # For Windows
    pytesseract.pytesseract.tesseract_cmd = r"F:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # For Linux/Cloud environments
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

st.set_page_config(page_title="AI Assistant App", layout="wide")

# Load BLIP Model for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# Load YOLO Model for object detection
@st.cache_resource
def load_yolo_model():
    model = YOLO("yolov8n.pt") 
    return model

yolo_model = load_yolo_model()

# Google Generative AI Configuration
chat_model = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyBpBRtvT1h6X9LIWWsrrQE9d9HAvBWUZEk",
    model="gemini-1.5-flash"
)
output_parser = StrOutputParser()

# Sidebar description
with st.sidebar:
    st.markdown("<h1 style='text-align: center; margin-top: 5px;'>👁️</h1>", unsafe_allow_html=True)
    st.markdown("""
    ### Features of the App:
    - **👁️ Scene Description**: Automatically generate meaningful context from uploaded images.
    - **🗣️ Text-to-Speech**: Convert extracted or generated text to engaging audio.
    - **🚧 Object Detection**: Identify obstacles or objects and provide safety tips.
    - **🤖 Personalized Assistance**: Tailored guidance based on scene analysis.

    Upload an image and select one of the options to experience the functionality!
    """)

# Streamlit app title
st.title("👁️🔊 AI Assistant for Visually Impaired Individuals")

# Upload an image
image_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    opencv_image = np.array(image)

    if opencv_image.shape[-1] == 4:  # RGBA to RGB
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGBA2RGB)
    elif len(opencv_image.shape) == 2 or opencv_image.shape[-1] == 1:  # Grayscale to RGB
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_GRAY2RGB)

# Inflect for pluralization
p = inflect.engine()

# Buttons in a row
st.markdown("<h3>Select a Feature:</h3>", unsafe_allow_html=True)
button_style = """
<style>
    .button-container {
        display: flex;
        justify-content: space-evenly;
        margin-top: 10px;
    }
    .button-container > div {
        flex: 1;
        text-align: center;
    }
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)
_, col1, col2, col3, col4, _ = st.columns([0.5, 1, 1, 1, 1, 0.5])

with col1:
    scene_btn = st.button("🖼️ Scene Description")
with col2:
    tts_btn = st.button("🗣️ Text to Speech")
with col3:
    detect_btn = st.button("🚧 Detect Objects")
with col4:
    assist_btn = st.button("🤖 Assistance")

# Display the relevant feature's content
if scene_btn:
    st.subheader("Scene Description")
    if image_file:
        inputs = processor(image, return_tensors="pt")
        outputs = model.generate(**inputs)
        scene_context = processor.decode(outputs[0], skip_special_tokens=True)

        # LangChain for scene description
        scene_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant providing scene descriptions for visually impaired users."),
            ("human", f"The scene context: {scene_context}. Provide a brief and clear description of the scene, "
                      "highlighting important details like objects, activities, and environment in short paragraphs of 2-3 sentences.")
        ])
        scene_chain = scene_prompt_template | chat_model | output_parser
        scene_description = scene_chain.invoke({})
        st.write(scene_description)

if tts_btn:
    st.subheader("Text-to-Speech")
    if image_file:
            # Extract text using OCR
            extracted_text = pytesseract.image_to_string(image).strip()
            if not extracted_text:
                inputs = processor(image, return_tensors="pt")
                outputs = model.generate(**inputs)
                extracted_text = processor.decode(outputs[0], skip_special_tokens=True)
                st.info(f"Generated Description: {extracted_text}")
            else:
                st.info(f"Extracted Text: {extracted_text}")

            # LangChain for enhanced text-to-speech
            speech_prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant converting text into audio for visually impaired users."),
                ("human", f"The extracted text: {extracted_text}. Convert this into a spoken-friendly version that is brief and engaging in atleast a short paragraphs of 2 or 3 clearly. Also do not mention this prompt as highlight which feels awkward fir the users.")
            ])
            speech_chain = speech_prompt_template | chat_model | output_parser
            enhanced_text = speech_chain.invoke({})

            st.write(enhanced_text,extracted_text)

            # Convert text to speech and play
            speech = pyttsx3.init()
            output_file = "enhanced_text_audio.mp3"
            speech.save_to_file(enhanced_text, output_file)
            speech.runAndWait()
            st.audio(output_file, format="audio/mp3")
            st.success("Text successfully converted to speech!")

if detect_btn:
    st.subheader("Object Detection")
    if image_file:
            # Detect objects using YOLO
            results = yolo_model(opencv_image)
            labeled_image = results[0].plot()
            st.image(labeled_image, caption="Object Detection Output", use_container_width=True)

            # Extract and count detected objects
            object_counts = {}
            for box in results[0].boxes.data.cpu().numpy():
                cls = int(box[5])  # Class index
                object_name = results[0].names[cls]
                object_counts[object_name] = object_counts.get(object_name, 0) + 1

            # Create natural language summary
            object_descriptions = [f"{count} {p.plural(obj, count)}" for obj, count in object_counts.items()]
            natural_language_output = ", ".join(object_descriptions)

            # LangChain for object significance
            object_prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant analyzing detected objects for visually impaired users."),
                ("human", f"Detected objects: {natural_language_output}. Provide a brief and clear explanation of these objects "
                        "and offer navigational tips for safety."
                        "Do not highlight the prompt mentioned here instead give reasonale ouput.")
            ])
            object_chain = object_prompt_template | chat_model | output_parser
            object_description = object_chain.invoke({})

            st.subheader("List of Objects or Obstacles")
            st.write("Detected objects:")
            st.text(natural_language_output)
            st.write("Description:")
            st.write(object_description)

            # Convert object description to speech
            speech = pyttsx3.init()
            output_file = "objects_audio.mp3"
            speech.save_to_file(object_description, output_file)
            speech.runAndWait()
            st.audio(output_file, format="audio/mp3")

if assist_btn:
    st.subheader("Personalized Assistance")
    if image_file:
            # Extract text using OCR or BLIP for fallback
            extracted_text = pytesseract.image_to_string(image).strip()
            if not extracted_text:
                inputs = processor(image, return_tensors="pt")
                outputs = model.generate(**inputs)
                extracted_text = processor.decode(outputs[0], skip_special_tokens=True)

            # LangChain for personalized assistance
            assistance_prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant offering tailored guidance for visually impaired users."),
                ("human", f"The uploaded image contains the following context: {extracted_text}. "
                        "Provide actionable suggestions in a clear and concise format to assist the user effectively and also do not highlight visually imapired such type of words  here as user needs only accurate output.")
            ])
            assistance_chain = assistance_prompt_template | chat_model | output_parser
            assistance_response = assistance_chain.invoke({})

            st.write(assistance_response)
            speech = pyttsx3.init()
            output_file = "personalized_assistance_audio.mp3"
            speech.save_to_file(assistance_response, output_file)
            speech.runAndWait()
            st.audio(output_file, format="audio/mp3")
