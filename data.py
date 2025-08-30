#%%
import kagglehub
import torch
import torch.nn as nn

from torchvision.transforms import v2

import os
import cv2
import cairosvg
from io import BytesIO
from PIL import Image
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("lantian773030/pokemonclassification")
print("Path to dataset files:", path)
#%%
class PokeData():  # for training/testing
    labels = ['Abra', 'Aerodactyl', 'Alakazam', 'Alolan Sandslash', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill',
                    'Bellsprout', 'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard',
                    'Charmander',
                    'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 'Dodrio',
                    'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 'Ekans', 'Electabuzz',
                    'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 'Flareon', 'Gastly', 'Gengar',
                    'Geodude',
                    'Gloom', 'Golbat', 'Goldeen', 'Golduck', 'Golem', 'Graveler', 'Grimer', 'Growlithe', 'Gyarados',
                    'Haunter', 'Hitmonchan', 'Hitmonlee', 'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx',
                    'Kabuto', 'Kabutops', 'Kadabra', 'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Krabby', 'Lapras',
                    'Lickitung', 'Machamp', 'Machoke', 'Machop', 'Magikarp', 'Magmar', 'Magnemite', 'Magneton',
                    'Mankey',
                    'Marowak', 'Meowth', 'Metapod', 'Mew', 'Mewtwo', 'Moltres', 'MrMime', 'Muk', 'Nidoking',
                    'Nidoqueen',
                    'Nidorina', 'Nidorino', 'Ninetales', 'Oddish', 'Omanyte', 'Omastar', 'Onix', 'Paras', 'Parasect',
                    'Persian', 'Pidgeot', 'Pidgeotto', 'Pidgey', 'Pikachu', 'Pinsir', 'Poliwag', 'Poliwhirl',
                    'Poliwrath',
                    'Ponyta', 'Porygon', 'Primeape', 'Psyduck', 'Raichu', 'Rapidash', 'Raticate', 'Rattata', 'Rhydon',
                    'Rhyhorn', 'Sandshrew', 'Sandslash', 'Scyther', 'Seadra', 'Seaking', 'Seel', 'Shellder', 'Slowbro',
                    'Slowpoke', 'Snorlax', 'Spearow', 'Squirtle', 'Starmie', 'Staryu', 'Tangela', 'Tauros', 'Tentacool',
                    'Tentacruel', 'Vaporeon', 'Venomoth', 'Venonat', 'Venusaur', 'Victreebel', 'Vileplume', 'Voltorb',
                    'Vulpix', 'Wartortle', 'Weedle', 'Weepinbell', 'Weezing', 'Wigglytuff', 'Zapdos', 'Zubat']
    
    def __init__(self, dataset):
        with open(f'./labels/{dataset}.txt', 'r') as f:
            self.img_labels = []
            self.img_files = []
            lines = f.readlines()

            for line in lines:
                self.img_files.append(line.split('\n')[0])
                self.img_labels.append(line.split('/')[-2])

        self.nimgs_original = len(self.img_labels)

        if dataset == 'train':
            for i in range(self.nimgs_original):
                self.img_files.append(self.img_files[i])
                self.img_labels.append(self.img_labels[i])

        self.transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((224, 224)),
            # v2.RandomHorizontalFlip(p=0.5),
            # v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
            v2.ToTensor()
        ])

        self.transform_hflip = v2.Compose([
            v2.ToPILImage(),
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]

        pimg = Path(self.img_files[idx])

        img_name = os.path.split(pimg)[-1]
        img_type = img_name.split(".")[-1]
        if img_type == 'svg':
            print('image is svg format - further processing necessary')

        img = cv2.imread(pimg) #by default, openCV is BGR but matplotlib is RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = torchvision.io.read_image(pimg)
        img = self.transform(img)
        # print('image successfully read')

        if idx > self.nimgs_original:
            img = self.transform_hflip(img)
        else:
            img = self.transform(img)

        #image needs to be of type float32 and label needs to be torch long (the label is an index, not name)
        return img, torch.tensor(self. labels.index(label), dtype=torch.long)

    def get_name(self, idx):
        return self.labels[idx]

#%%
def genannot(split, path):
    import random
    from os import listdir
    
    if not os.path.exists(f'./labels/'):
        os.makedirs(f'./labels/')

    ftrain = open('./labels/train.txt','w')
    fval = open('./labels/val.txt','w')
    ftest = open('./labels/test.txt','w')
    fall = open('./labels/all.txt','w')
    newline = '\n'

    for label in listdir(path):
        imgs = os.listdir(path + f'/{label}')
        random.shuffle(imgs)

        totalval = int(split[1]*len(imgs))
        totaltest = int(split[2]*len(imgs))
        totaltrain = len(imgs) - totalval - totaltest

        for i, img in enumerate(imgs):
            fall.write(path + f'/{label}/{img}{newline}')
            if i < totaltrain:
                ftrain.write(path + f'/{label}/{img}{newline}')
            elif i >= totaltrain and i < (len(imgs)-totaltest):
                fval.write(path + f'/{label}/{img}{newline}')
            else:
                ftest.write(path + f'/{label}/{img}{newline}')

    ftrain.close
    fval.close
    ftest.close
    fall.close

#%%
if __name__ == "__main__":
    split = [0.7, 0.15, 0.15] #70% training, 15% validation, 15% test 
    genannot(split, path + '/PokemonData')