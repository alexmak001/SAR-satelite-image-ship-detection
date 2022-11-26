import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
from torchvision.datasets import ImageFolder
# import os
from torchvision import models, transforms
import torch.optim as optim
import shutil
import os

# move data
os.system('mkdir data/image_data')
os.system('mkdir data/image_data/offshore')
os.system('mkdir data/image_data/inshore')
with open("data/main/test_offshore.txt") as f:
   while True:
       line =  f.readline()
      line = line.replace("\n","")
       filename = line + ".jpg"

       original = os.path.abspath("data/test/images/" + filename)
        
       target = os.path.abspath("data/image_data/offshore/" + filename)

       try:
           shutil.move(original, target)
       except Exception as e:
           print(e)
            print(filename + " NOT FOUND - offshore")

       if not line:
           break


with open("data/main/test_inshore.txt") as f:
   while True:
       line =  f.readline()
       line = line.replace("\n","")
       filename = line + ".jpg"

       original = os.path.abspath("data/test/images/" + filename)
        
       target = os.path.abspath("data/image_data/inshore/" + filename)

       try:
           shutil.move(original, target)
       except Exception as e:
           print(e)
            print(filename + " NOT FOUND - inshore")

       if not line:
           break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Compose([transforms.Resize((32,32))])])

dataset = ImageFolder(os.path.abspath('data/image_data'),transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [2100,900])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False)

dataloaders = [train_loader, test_loader]

classes = os.listdir('data/image_data')

model = models.resnet50(pretrained=True).to(device)

for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, 2)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())


def train_model(model, criterion, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in [0,1]:
            if phase == 0:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
    return model

model_trained = train_model(model, criterion, optimizer, num_epochs=50)

torch.save(model_trained.state_dict(), 'offshore_inshore_clf_50e.h5')
