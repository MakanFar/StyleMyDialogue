from StyleMyDialogue import StyleMyDialogue
from data_processor import get_characters, get_dialouges, get_images, get_emotions
import yaml
import csv
from dotenv import load_dotenv
import os

def confiqure():
    load_dotenv()

def main():
    
    confiqure()

    with open('config.yml', 'r') as file:
        vars = yaml.safe_load(file)

    model = StyleMyDialogue(get_characters(), get_emotions(), os.getenv('api_key'))
    chars,emotions = model.similarities(get_images())
    dialogues = get_dialouges()
    results = model.generate_prompts(chars,emotions,dialogues)
    results['generic dialogue'] = dialogues
    print(results)

    with open("results.csv", "w") as outfile:
        
        writerfile = csv.writer(outfile) 
        writerfile.writerow(results.keys())
        writerfile.writerows(zip(*results.values()))
        

if __name__ == "__main__":
    main()

    


