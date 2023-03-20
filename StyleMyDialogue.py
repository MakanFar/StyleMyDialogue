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

    def __init__(self, characters, emotions, chatgpt_key):

        self.chatgpt_key = chatgpt_key
        self.characters= characters
        self.emotions= emotions

        
    def similarities(self, img_path):
        model, preprocess = clip.load("ViT-B/32")
        model.to(device).eval()

        images = []
        for im in img_path: 
            response = requests.get(im)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(preprocess(image))

        image_input = torch.tensor(np.stack(images)).to(device)
        char_tokens = clip.tokenize(["This is a" + ch for ch in self.characters]).to(device)
        emotion_tokens = clip.tokenize(["This character is" + em for em in self.emotions]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input).float()
            char_features = model.encode_text(char_tokens).float()
            emotion_features = model.encode_text(emotion_tokens).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        char_features /= char_features.norm(dim=-1, keepdim=True)
        emotion_features /= emotion_features.norm(dim=-1, keepdim=True)

        char_similarity = char_features.cpu().numpy() @ image_features.cpu().numpy().T
        emotion_similarity = emotion_features.cpu().numpy() @ image_features.cpu().numpy().T
        return char_similarity,emotion_similarity

        
    def generate_prompts(self, char, emotion, dialogues):

        openai.api_key = self.chatgpt_key

        char_predictions = torch.topk(torch.tensor(char), 1, dim=0)[1][0]
        char_predictions = char_predictions.numpy()
        pred_characters = [self.characters[i] for i in char_predictions]
        print(pred_characters)

        emotion_predictions = torch.topk(torch.tensor(emotion), 1, dim=0)[1][0]
        emotion_predictions = emotion_predictions.numpy()
        pred_emotions = [self.emotions[i] for i in emotion_predictions]
        print(pred_emotions)

   
        results_dict= {}
        

        for character, emotion  in zip (pred_characters,pred_emotions): 
            character_dais = []
            for dai in dialogues:

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                      
                            {
                                "role": "user",
                                "content": "Keep in mind that maximum length is 20 words, rewrite the following dialogue in the style of a {} {}: Original Dialogue: {} Rewritten Dialogue:".format(emotion,character,dai)
                            }
                        ] 
                    )

                result = ''
                for choice in response.choices:
                    result += choice.message.content
                character_dais.append(result)
            
            results_dict[emotion+" "+character]=character_dais

        return results_dict
