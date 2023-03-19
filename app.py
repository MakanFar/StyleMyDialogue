import streamlit as st
import clip
import numpy as np
import streamlit as st
import openai
import torch
from PIL import Image
import openai
import yaml
from data_processor import get_characters

with open('config.yml', 'r') as file:
    vars = yaml.safe_load(file)


def classify_image(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32")
    model.to(device)
    model.eval()
    
    characters = get_characters()

    image = Image.open(image_path).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = preprocess(image)
    images=[]
    images.append(image)
    image_input = torch.tensor(np.stack(images)).to(device)
    text_tokens = clip.tokenize(["This is " + ch for ch in characters]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
    predictions = torch.topk(torch.tensor(similarity), 1, dim=0)[1][0]
    predictions = predictions.numpy()
    for i in predictions:
        character = characters[i]
    return character


def style_dialouge(character, dialouge):

    openai.api_key = st.secrets["api_key"]


  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {
                "role": "system",
                "content": "Pretend to be a {}.".format(character)
                },
                {
                    "role": "user",
                    "content": "Keep in mind that the maximum length is 20 words. Now, say this pharase: {}".format(dialouge)
                }
            ])
        

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

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
        image_features = classify_image(image_file)
        dialogue = style_dialouge(image_features, prompt)
        st.write("Here's the stylized dialogue:")
        st.write(dialogue)

if __name__ == "__main__":
    main()
