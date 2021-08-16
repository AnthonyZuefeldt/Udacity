# ------------------------------------------------------------------------------------------------------------
#                   IMPORTED PACKAGES

# List of imported packages

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
    parser = argparse.ArgumentParser(description = "Train.py")
    parser.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu")
    parser.add_argument('--arch', default = "vgg16", help = 'Choose a model: vg 16 or vg19')
    parser.add_argument('--save_dir', dest = "save_dir", action = "store", default = "./checkpoint.pth")
    parser.add_argument('--learning_rate', dest = "learning_rate", action = "store", default = 0.001)
    parser.add_argument('--hidden_units', type = int, dest = "hidden_units", action = "store", default = 120)
    parser.add_argument('--epochs', dest = "epochs", action = "store", type = int, default = 1)
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
#                   DIRECTORIES

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# ------------------------------------------------------------------------------------------------------------
#                   IMAGE TRANSFORMS

#1 Converts images into tensors that are used for this model
#2 Defines the transforms for the training, validation, and testing sets
#3 Transforms are used to add variability in the training images set to help this model generalize images

    #transform functions:
        #transforms.Compose chains together multiple transform functions
        #transforms.RandomResizedCrop crops the images to a specified pixel size
        #transforms.RandomHorizontalFlip flips the images horizontally
        #transforms.RandomVerticalFlip flips the images vertically
        #transforms.CenterCrop crops images to a standard size, centered on the original image
        #transforms.ToTensor converts the image to tensor
        #transforms.Normalize scales image color range from 0-255 to 0-1

#4 Transform functions in the validation and testing sets are establish image compatibility with model/code      

data_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])
}

#1 Load the datasets with ImageFolder, creating a dataset object that holds the images

dataset_images = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}


# ------------------------------------------------------------------------------------------------------------
#                           DATA LOADING

#1 Defines the dataloaders with the image datasets and the trainforms
#2 batch_size is a hyperparameter controlling the number of samples to process before internal parameters are updated

    # this dataset contains 6552 training samples, 818 validation samples, and 819 testing samples
    # batch_size = sample size / batch count

#3 Shuffle function acts as a permutation tool, activating RnadomSampler to shuffle indices of data

dataset_train = torch.utils.data.DataLoader(dataset_images['train'], batch_size=64, shuffle=True)
dataset_validation = torch.utils.data.DataLoader(dataset_images['valid'], batch_size =64,shuffle = True)
dataset_test = torch.utils.data.DataLoader(dataset_images['test'], batch_size = 64, shuffle = True)

dataloaders = [dataset_train, dataset_validation, dataset_test]

#1 Print the dataset sizes
dataset_sizes = {x: len(dataset_images[x]) for x in ['train', 'valid', 'test']}
print(dataset_sizes)

# ------------------------------------------------------------------------------------------------------------
#                                   LABEL MAPPING

#import json object 'cat_to_name' for dictionary mapping the integer encoded categories to actual flower names
# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

category_count = len(cat_to_name)

cat_to_name

print(category_count)

# ------------------------------------------------------------------------------------------------------------
#                               BUILDING THE CLASSIFIER

#1 Building and training the network
#2 Pretrained_model variable used to hold model

pretrained_model = args.arch

# conditional state to allow for toggle between pretrained model vgg16 and vgg19

if pretrained_model == 'vgg16':
    model = models.vgg16(pretrained=True)
    input_count = 25088
elif pretrained_model == 'vgg19':
    model = models.vgg19(pretrained=True)
    input_count = 25088
else:
    print('Architecture does not currently support model', pretrained_model, 'try vgg16 or vgg19 instead.')
    
print(model)

#1 Freeze convolutional layers so that they remain static

for param in model.parameters():
    param.requires_grad = False
    
#1 Define and set the input values of the connected layer
#2 OrderDict preserves the order of the keys

classifier = nn.Sequential(OrderedDict([
            ('inputs', nn.Linear(input_count, args.hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout',nn.Dropout(0.45)),
            ('hidden_layer1', nn.Linear(args.hidden_units, int(0.75 * args.hidden_units))),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(int(0.75 * args.hidden_units),int(0.60 * args.hidden_units))),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(int(0.60 * args.hidden_units),category_count)),
            ('output', nn.LogSoftmax(dim=1))]))
   
model.classifier = classifier  

# ------------------------------------------------------------------------------------------------------------
#                               HYPER PARAMETERS

#1 Hyper parameters and inputs for training:

    # Criterion 
    # Epoch_count
    # Optimizer
    
criterion = nn.NLLLoss()
epoch_count = args.epochs
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

#1 Placeholder parameters for loss and accuracy

training_loss = 0.0
training_accuracy = 0.0
        
validation_loss = 0.0
validation_accuracy = 0.0

# ------------------------------------------------------------------------------------------------------------
#                                   TRAINING THE CLASSIFIER

#Total loss and accuracy is computed for the whole batch, 
#which is then averaged over all the batches to get the loss and accuracy values for the whole epoch.

with active_session():
    for epoch in range(epoch_count):
        print("Epoch: {}/{}".format(epoch+1, epoch_count))
   
        # Activates training mode
        model.train()
        model.to(device)

        for i, (inputs, labels) in enumerate(dataset_train):
            
            # Defines the CPU or GPU operator
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset optimizer gradients
            optimizer.zero_grad()

            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backpropagate the gradients
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Compute the total loss for the batch and add it to training_loss
            training_loss += loss.item() * inputs.size(0)

            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            # Convert correct_counts to float and then compute the mean
            accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
            accuracy_percentage = accuracy * 100
                        

            # Compute total accuracy in the whole batch and add to training_accuracy
            training_accuracy += accuracy.item() * inputs.size(0)

            print("Batch {:02d} | Training Loss: {:.4f} | Accuracy: {:.3f}%".format(i, loss.item(), accuracy_percentage.item()))
  

# ------------------------------------------------------------------------------------------------------------
#                                           VALIDATION

#1 Validate the test set!

with torch.no_grad():
 
    model.eval()

    # Validation loop
    for batch, (inputs, labels) in enumerate(dataset_validation):
        
        inputs, labels = inputs.to(device), labels.to(device)
 
        # Compute forward and backwards pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
 
        # Compute total loss per batch added to validation loss
        validation_loss += inputs.size(0) * loss.item()
 
        # Calculate validation accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
 
        # Calculate the mean accuracy
        model_accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
        
        accuracy_percentage = model_accuracy * 100
 
        # Total accuracy of whole batch added to validation accuracy
        validation_accuracy += model_accuracy.item() * inputs.size(0)
 
        print("Test Number: {:02d} | Validation Accuracy: {:.2f}%".format(batch, accuracy_percentage.item()))

# ------------------------------------------------------------------------------------------------------------
#                                   TEST

# Evaluation
test_accuracy = 0
for images,labels in dataset_test:
    model.eval()
    images,labels = images.to(device),labels.to(device)
    log_ps = model.forward(images)
    ps = torch.exp(log_ps)
    top_ps,top_class = ps.topk(1,dim=1)
    matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
    accuracy = matches.mean()
    test_accuracy += accuracy
print(f'Model Test Accuracy: {test_accuracy/len(dataset_test)*100:.2f}%')

# ------------------------------------------------------------------------------------------------------------
#                                       SAVE THE CHECKPOINT

#1 Record the mapping predicted class and class name
model.class_to_idx = dataset_images['train'].class_to_idx

#1 Create the checkpoint dictionary containing specific variables
checkpoint = {'batch_size' : 64,
              'epoch_count' : epoch_count,
              'classifier' : model.classifier,
              'model_architecture' : pretrained_model,
              'mapping' : model.class_to_idx,
              'optimizer' : optimizer.state_dict(),
              'model_state' : model.state_dict(),
              'class_to_idx' : model.class_to_idx
             }
#1 Save the checkpoint
torch.save(checkpoint, args.save_dir)
