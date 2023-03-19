import os
import random
import csv
import yaml

with open('config.yml', 'r') as file:
    vars = yaml.safe_load(file)



def get_dialouges():

    dialogue_list = vars['dialogues']

    if vars['dialogues']=='data/dialogues_text.txt':

        path = os.path.join(vars['dialogues'])
        with open(path, 'r') as f:
            conversations = [line.split(' __eou__')[:-1] for line in f]
        
        dialogue_list = conversations[random.randint(0,len(conversations)-1)][:2]

    return dialogue_list

def get_characters():

    csv_file = open(vars['characters'])
    csv_reader = csv.DictReader(csv_file)

    characters = []

    for row in csv_reader:
        char = row['Character']
        characters.append(char)

    return characters

def get_images():

    csv_file = open(vars['images'])
    csv_reader = csv.DictReader(csv_file)

    addresses = []

    for row in csv_reader:
        address = row['Address']
        addresses.append(address)

    return addresses