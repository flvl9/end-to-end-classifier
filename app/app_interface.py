import json
import streamlit as st
from app_utils import *

models_info = retrieve_models_info()

if not models_info["names"]:
    st.info("There are no models available. Please train and deploy a model first.")
    st.stop()

models_info["traffics"] = normalize_traffic(models_info["traffics"])
selected_model_data = select_model(models_info)
with open("app/label_info.json", "r") as file:
    label_and_treatment = json.load(file)
treatment = None

st.title("Plant Disease Detector")
with st.sidebar:
    st.header("Model Info")
    st.write(f"Name: {selected_model_data["name"]}")
    st.metric("Version", selected_model_data["version"])
    st.metric("Traffic", selected_model_data["traffic"])
    st.progress(selected_model_data["score"], text=f"Validation F1 Score: {selected_model_data["score"]}")
    
col1, col2 = st.columns([1,1])

with col1:
    with st.container(border=True):
        st.subheader("Your Image", text_alignment="center")
        image_file = st.file_uploader(
            "Please select an image",
            type="image/*",
            accept_multiple_files=False
        )
        if image_file is not None:
            st.image(
                image_file,
                caption="Your image"
            )      

with col2: 
    st.subheader("Result Card", text_alignment="center")
    if image_file is not None:
        img_tensor = preprocess_image(image_file)
        clas, score = make_prediction(selected_model_data["model"], img_tensor)
        raw_label = label_and_treatment["labels"][f"{clas}"]
        crop, disease = format_label(raw_label)
        raw_treatment = label_and_treatment["treatments"][raw_label]
        treatment = format_treatment(raw_treatment)
        st.write(f"Crop: {crop}")
        st.write(f"Disease: {disease}")
        st.progress(score, text=f"Score: {score}")
        
if treatment is not None:
    with st.expander("📋 Treatment"):
        for item in treatment:
            st.markdown(f"*{item}")
        st.warning("Note: If the disease has no cure, the other suggestions should be taken" \
        "to prevent the disease from spreading to other helthy plants.")
