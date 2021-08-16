import argparse #parser for command-line interface

from collections import OrderedDict #container datatype tool
import json #adds JSON functionality
import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image

import seaborn as sns

import time

import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

from workspace_utils import active_session

# ------------------------------------------------------------------------------------------------------------
#                   ARG PARSER DEFINED

def arg_parser():
    parser = argparse.ArgumentParser(description = 'predict.py')
    parser.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'gpu')
    parser.add_argument('--arch', default = "vgg16", help = 'Choose a model: vg 16 or vg19')
    parser.add_argument('--checkpoint', dest = 'checkpoint', action = 'store', default = './checkpoint.pth')
    parser.add_argument('--k_values', dest = 'k_values', action = 'store', type = int, default = 5)
    parser.add_argument('--map_category', dest = 'map_category', action = 'store', default = 'cat_to_name.json')
    parser.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.001)
    parser.add_argument('--hidden_units', type = int, dest = 'hidden_units', action = 'store', default = 120)
    parser.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 12)
    parser.add_argument('--image_dir', dest = 'image_dir', default = './flowers/train')
    args = parser.parse_args()
    return args

args = arg_parser()

# ------------------------------------------------------------------------------------------------------------
#                   GPU TOGGLE

# check if GPU is available
# writes model computing to device variable
def check_gpu(gpu_arg):
    if not gpu_arg:
        print('Graphical Processing Unit (GPU) active.')
        return torch.device("cpu")
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    if device == "cpu":
        print('Central Processing Unit (CPU) active.')
    return device

device = check_gpu(args.gpu)

print(device)

# ------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------
#                       LOAD THE CHECKPOINT
              
# Loads a checkpoint and rebuilds the model
def checkpoint_data(file_path):
    checkpoint = torch.load(file_path)
    model = checkpoint['model_architecture']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['mapping']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

model = checkpoint_data('checkpoint.pth')
print(model)

# ------------------------------------------------------------------------------------------------------------
#                                       IMAGE PREPROCESSING

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    images = Image.open(image).convert('RGB')
    images.thumbnail(size=(256,256))
    existing_width, existing_height = images.size
    
    new_width, new_height = 224, 224
    left = (existing_width - new_width)/2
    top = (existing_height - new_height)/2
    right = (existing_width + new_width)/2
    bottom = (existing_height + new_height)/2
    images = images.crop((left, top, right, bottom))
    
    image_to_tensor = transforms.ToTensor()
    image_normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    processed_image = image_normalize(image_to_tensor(images))
        
    numpy_image_array = np.array(processed_image)
    return numpy_image_array

# ------------------------------------------------------------------------------------------------------------
#                               CLASS PREDICTION

# 1 Define three primary data points: label data, prediction method, and image probabilities predictions
# 2 Label data is extracted from JSON data and organized into a dictionary
# 3 Predictions uses the image classifier model built in this code to map the class of the flower image
# 4 Print the top flower class predictions and their estimated probabilities

class_to_idx = dataset_images['train'].class_to_idx

indices = dict(map(reversed, class_to_idx.items()))

#1 Label Data function
def label_data(file, flower_classes):
    
    with open (file, 'r') as json_data:
        flower_data = json.load(json_data)
    labels = []
    for flower_type in flower_classes:
        labels.append(flower_data[flower_type])
        
    return (labels)

#1 Prediction function
def predict(image_filepath, model, indices, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.to(device)
    model.eval()
    
    images = process_image(image_filepath) # call the image preprocessing function and the flower image file
    images = torch.from_numpy(images) #make it a tensor from a numpy array!
    images = torch.unsqueeze(images, 0).to(device).float() 
    
    exponential_input = model.forward(images)
    exponential = torch.exp(exponential_input)
    probability_percentage = exponential * 100
    
    top_k_value, top_idx_value = exponential.topk(topk, dim=1)
    list_k_values = top_k_value.tolist()[0]
    list_idx_values = top_idx_value.tolist()[0]
    
    flower_classes = []
    
    model.train()
    
    for idx in list_idx_values:
        flower_classes.append(indices[idx])
    return list_k_values, flower_classes

#1 Image probabilities function for printing probabilities / potential flower matches
def image_probabilities(probabilities, flower_classes, image, category_names = None):
    
    print('Flower Image File:', image)
    
    if category_names:
        labels = label_data (category_names, flower_classes)
        for each, (exponential, ls, cs) in enumerate(zip(probabilities, labels, flower_classes), 1):
            print(f'{each}) {ls.title()} | Classification Number {cs} | Probability {exponential*100:.2f}%')
    else:
        for each, (exponential, cs) in enumerate(zip(probabilities, flower_classes),1):
            print(f'{each}) {exponential*100:.2f}% Classification Number {cs} ')
    print('')

probabilities, flower_classes = predict(image_path, model, indices, arg.k_values, device)

image_probabilities(probabilities, flower_classes, image_path.split('/')[-1],'cat_to_name.json')