# load libraries
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from os import listdir
import time
import copy
import argparse

# Initiate default variables
arch = 'vgg16'
learning_rate = 0.001
epochs = 4
print_every=25
steps=0
model= models.vgg16(pretrained=True)


# Directory location of images
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str, help='Location of directory')
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model')
parser.add_argument('-e','--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent')
parser.add_argument('-s','--save_dir',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available')

args = parser.parse_args()

# Select parameters entered in command line

if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model= models.vgg16(pretrained=True) 
for param in model.parameters():
    param.requires_grad=False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(4096,1000,bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier=classifier

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

for param in model.parameters():
    param.requires_grad=False  
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(4096,1000,bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier=classifier
    criterion = nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=0.0005)
    
model.to('cuda')

def validation(model, vloader, criterion):
    vloss = 0
    accuracy = 0
    for ii, (images, labels) in enumerate(vloader):
    
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        vloss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()  
    return vloss, accuracy

def testing(correct, total):
    correct = 0
    total = 0
    model.to('cuda:0')

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the model is: %d %%' % (100 * correct / total))

for e in range(epochs):
    running_loss =0
    for ii, (images, labels) in enumerate(trainloader):
         steps += 1
         images, labels = images.to('cuda'), labels.to('cuda')
         optimizer.zero_grad()
         outputs = model.forward(images)
         loss = criterion(outputs, labels)
         loss.backward()
         optimizer.step()
         running_loss += loss.item()
         if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    vloss, accuracy = validation(model, vloader, criterion)

                print("Epoch:{}/{}".format(e+1,epochs),
                "Training Loss:{:.4f}".format(running_loss/print_every),
                "Valid Loss:{:.4f}".format(loss),
                "Valid Accuracy:{:.4f}".format(accuracy))

                running_loss = 0
                
                model.train()
                
print("COMPLETE")

