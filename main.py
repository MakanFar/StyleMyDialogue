from StyleMyDialogue import StyleMyDialogue
from data_processor import get_characters, get_dialouges, get_images
import yaml
import csv


def main():

    with open('config.yml', 'r') as file:
        vars = yaml.safe_load(file)

    model = StyleMyDialogue(get_characters(),vars['api_key'])
    chars = model.character_similarities(get_images())
    dialogues = get_dialouges()
    results = model.generate_prompts(chars,dialogues)
    results['generic dialogue'] = dialogues
    print(results)

    with open("results.csv", "w") as outfile:
        
        writerfile = csv.writer(outfile) 
        writerfile.writerow(results.keys())
        writerfile.writerows(zip(*results.values()))
        

if __name__ == "__main__":
    main()

    


