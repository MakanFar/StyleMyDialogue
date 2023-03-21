import streamlit as st
import clip
import numpy as np
import streamlit as st
import openai
import torch
from PIL import Image
import openai
import yaml
from data_processor import get_characters, get_emotions


def classify_image(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32")
    model.to(device)
    model.eval()
    
    characters = get_characters()
    emotions = get_emotions()

    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = preprocess(image)
    images=[]
    images.append(image)
    image_input = torch.tensor(np.stack(images)).to(device)
    char_tokens = clip.tokenize(["This is an image of a " + ch for ch in characters]).to(device)
    emotion_tokens = clip.tokenize(["An image of a character that is emotionally " + em for em in emotions]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        char_features = model.encode_text(char_tokens).float()
        emotion_features = model.encode_text(emotion_tokens).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    char_features /= char_features.norm(dim=-1, keepdim=True)
    char_similarity = char_features.cpu().numpy() @ image_features.cpu().numpy().T
    emotion_features /= emotion_features.norm(dim=-1, keepdim=True)
    emotion_similarity = emotion_features.cpu().numpy() @ image_features.cpu().numpy().T

    char_predictions = torch.topk(torch.tensor(char_similarity), 1, dim=0)[1][0]
    char_predictions = char_predictions.numpy()
    emotion_predictions = torch.topk(torch.tensor(emotion_similarity), 1, dim=0)[1][0]
    emotion_predictions = emotion_predictions.numpy()

    for i,j in zip(char_predictions,emotion_predictions):
        character = characters[i]
        emotion = emotions[j]

    return character, emotion


def style_dialouge(character, emotion, dialouge):

    openai.api_key = st.session_state["api"]


  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
             {
                "role": "user",
                "content": "Keep in mind that maximum length is 20 words, rewrite the following dialogue in the style of a {} {}: Original Dialogue: {} Rewritten Dialogue:".format(emotion,character,dialouge)
            }
            ])
        

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def login():
    
    st.write("Welcome to the login screen")
    api = st.text_input("Please enter your OpenAI API key here")
    global proceed
    if st.button("Proceed") and api:
        st.session_state["page"] = "main"
        st.session_state["api"] = api
        st.balloons()
        st.experimental_rerun()
    
    else:
        st.error("Api key not entered")

    return api
    


def main():
    st.set_page_config(page_title="Character Dialogue Styler", page_icon=":smiley:")
    # Add custom CSS styles
    st.markdown(
        """
        <style>
        .title {
            font-size: 3rem;
            font-weight: bold;
            color: #0072B1;
        }
        .app {
            background-color: #F5F5F5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Add app title and description
    st.markdown('<h1 class="title">Welcome to Character Dialogue Styler!</h1>', unsafe_allow_html=True)
    st.write("Upload an image of a character and enter a dialogue prompt to see the stylized output.")

    # Add file uploader and text input
    image_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Enter a dialogue for the character to say")

    # Display stylized output
    if image_file and prompt:
        character, emotion = classify_image(image_file)
        dialogue = style_dialouge(character, emotion, prompt)
        st.write("Here's the stylized dialogue:")
        st.write(dialogue)

if __name__ == "__main__":

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["page"] == "login":
        login()
    elif st.session_state["page"] == "main":
        main()


