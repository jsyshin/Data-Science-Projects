#load libraries
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from os import listdir
import json
import argparse
import matplotlib.pyplot as plt
import PIL

# Set default values
checkpoint = 'checkpoint.pth'
filepath =  'flowers/test/1/image_06743.jpg'
epochs=4
print_every=25
steps=0
topk=5

# Parameter entry in command line
parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint', action='store',type=str, help='Name of the model that needs to be loaded.')
parser.add_argument('-i','--image_path',action='store',type=str, help='Location of image')
parser.add_argument('-j', '--json', action='store',type=str, help='Name of Jason file')
parser.add_argument('-g','--gpu', action='store_true', help='GPU')

args = parser.parse_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def load_model(filepath):
    checkpoint = torch.load(filepath)
    
    model= models.vgg16(pretrained=True)
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096, bias=True)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.05)),
                          ('fc2', nn.Linear(4096,1000,bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier=classifier
    
    return model

model = load_model('checkpoint.pth') 

def process_image(image):
    pil_image = PIL.Image.open(image)
    adj = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    adj_image= adj(pil_image)
    array_adj_image=np.array(adj_image)
    return array_adj_image

def predict(image_path, model, topk=5): 
        model = model
        model.eval()
        model.to('cpu')
        model.double()
        img = process_image(image_path)
        img = torch.from_numpy(img).type (torch.DoubleTensor)
        img = img.unsqueeze_(0)

        with torch.no_grad ():
            output = model.forward (img)
        output= torch.exp (output)
        
        ps = torch.exp(output)
        top_ps = ps.topk(topk)
        probs = top_ps[0]
        probs = probs[0].numpy()
        classes = top_ps[1][0].numpy().tolist()
        return probs, classes
   
print(model)
input("Enter to continue to the prediction.")

probs, classes = predict('flowers/test/1/image_06743.jpg',model)

print(probs,classes)
