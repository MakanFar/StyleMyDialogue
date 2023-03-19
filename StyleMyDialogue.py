import requests
import json
import clip
from PIL import Image
import torch
import numpy as np
import openai
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"



class StyleMyDialogue:

    def __init__(self, characters, chatgpt_key):

        self.chatgpt_key = chatgpt_key
        self.characters= characters

        
    def character_similarities(self, img_path):
        model, preprocess = clip.load("ViT-B/32")
        model.to(device).eval()

        images = []
        for im in img_path: 
            response = requests.get(im)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(preprocess(image))

        image_input = torch.tensor(np.stack(images)).to(device)
        text_tokens = clip.tokenize(["This is " + ch for ch in self.characters]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            text_features = model.encode_text(text_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy() @ image_features.cpu().numpy().T

        
    def generate_prompts(self, similarities, dialogues):

        openai.api_key = self.chatgpt_key

        predictions = torch.topk(torch.tensor(similarities), 1, dim=0)[1][0]
        predictions = predictions.numpy()
        pred_characters = [self.characters[i] for i in predictions]
        print(pred_characters)

        ch_list = []
        dialogue_list=[]

        for ch in pred_characters: 
            prompt = "Pretend to be a {}.".format(ch)
            ch_list.append(prompt)

        for dai in dialogues:
            dialogue = "Say this pharase: {}".format(dai)
            dialogue_list.append(dialogue)


        results_dict= {}
        

        for character  in pred_characters: 
            character_dais = []
            for dai in dialogues:

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                            {
                            "role": "system",
                            "content": "Pretend to be a {}.".format(character)
                            },
                            {
                                "role": "user",
                                "content": "Keep in mind that the maximum length is 20 words. Now, say this pharase: {}".format(dai)
                            }
                        ]
                    )

                result = ''
                for choice in response.choices:
                    result += choice.message.content
                character_dais.append(result)
            
            results_dict[character]=character_dais

        return results_dict
