# create_class_names.py (run locally)
from torchvision import datasets
import json
TRAIN_DIR = "../data/Train"
class_names = datasets.ImageFolder(TRAIN_DIR).classes
json.dump(class_names, open("class_names.json","w"), indent=2)
