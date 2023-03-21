from StyleMyDialogue import StyleMyDialogue
from data_processor import get_characters, get_dialouges, get_images, get_emotions
import yaml
import csv
from dotenv import load_dotenv
import os
import numpy as np
import openai
import time

def confiqure():
    load_dotenv()


prompt = \
'''
Imagine that you are writing dialogues for different characters in a game.
You will be provided with a character type and a piece of dialogue.
Your job is to re-write the piece of dialogue as if the given character is saying it.
You should not significantly change the meaning of the original piece of dialogue.

Here's an example >
Character type: Viking
Piece of dialogue: I'm going to buy a toyota corrola.
Re-written dialogue: I'm gonna lay me coin on a Toyota Corolla!

Actual Task >
Character type: {}
Piece of dialogue: {}
Re-written dialogue:
'''

def generate_output(api_key, char, dialogue):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt.format(char, dialogue),
        }])

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result
    

def main():
    
    confiqure()
    np.random.seed(1213)

    with open('config.yml', 'r') as file:
        vars = yaml.safe_load(file)

    api_key = vars['api_key']
    characters = [
        'Cowboy',
        'Viking',
        'Vampire',
        'Monk',
        'Mr. Bean',
        'Snoop Dogg',
        'Pirate',
        'Napoleon Bonaparte',
        'Django',
        'A farmer in Idaho',
    ]

    path = os.path.join(vars['dialogues'])
    with open(path, 'r') as f:
        dialogues = [line.strip().split(' __eou__')[:-1] for line in f]
        dialogues = [[x.strip() for x in c]for c in dialogues]

    ll = []
    for d in dialogues:
        for c in d:
            ll.append(len(c))
    ll = np.array(ll)
    print(ll.mean())

    dialogue_lens = np.array([len(x) for x in dialogues]) 
    inds = np.where(dialogue_lens == 5)[0]
    np.random.shuffle(inds)
    styl_dialogues = []
    for i in inds[:50]:
        cur_styl_dialogue = []
        ci1, ci2 = np.random.permutation(len(characters))[:2]
        chars = [characters[ci1], characters[ci2]]
        print('='*40 + f' Original Dialg {i:3d} ' + '='*40)
        for j, cur_dialogue in enumerate(dialogues[i]):
            print(f'> ({j:2d}) [[ {j%2} ]]')
            print(f'>>>>> {cur_dialogue}')
        print('='*40 + f' Stylized Dialg {i:3d} ' + '='*40)
        for j, cur_dialogue in enumerate(dialogues[i]):
            cur_styl_dialogue.append(generate_output(api_key, chars[j%2], cur_dialogue))
            print(f'> ({j:2d}) {chars[j%2]}')
            print(f'>>>>> {cur_styl_dialogue[-1]}')
        print('='*100 + '\n\n')
        time.sleep(20)


if __name__ == "__main__":
    main()

