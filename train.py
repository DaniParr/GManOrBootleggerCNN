import torch
from PIL import Image

from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.models as models

#Testing labeling of dataset
import matplotlib.pyplot as plt
import numpy as np

class TrainBinaryVisionModel:

    def __init__(self):
        #Get data
        #self.train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
        self.dataset = self.getDataset()

        #NeuroNetwork
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clf = ImageClassifier().to(self.device)

        #Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        #Optimizer
        self.opt = Adam(self.clf.parameters(), lr = 1e-3)

        #Create a split in the data for: Training, Validation, Testing
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset, [self.train_size, self.val_size])
        
        # Create DataLoaders
        self.batch_size = 32
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)




    def getDataset(self):
        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size (e.g., 224x224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

        # Specify the path to the dataset root folder
        data_path = 'imgData'

        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        return dataset
    
    def validateLabels(self)->DataLoader:
        # G-Man:      1
        # Bootlegger: 0

        transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size (e.g., 224x224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

        # Specify the path to the dataset root folder
        data_path = 'imgData'

        # Create the ImageFolder dataset
        dataset = datasets.ImageFolder(root=data_path, transform=transform)

        # Create a DataLoader to efficiently load and batch the data
        batch_size = 32
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Iterate over the data_loader to get batches of images and their corresponding labels
        for images, labels in data_loader:
            print(labels)
            print(type(images[0]))
            image = images[0].numpy().transpose((1, 2, 0))
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = image * std + mean
            image = np.clip(image, 0, 1)
            plt.imshow(image)
            plt.title(f"{labels[0]} Dimensions: {image.shape}")
            plt.show()
        
        return data_loader


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # Adjusted fully connected layer input size
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    training = TrainBinaryVisionModel()
    # training.validateLabels()

# Training loop
    # num_epochs = 20

    # for epoch in range(num_epochs):
    #     for batch in training.train_loader:
    #         x,y = batch
    #         x,y = x.to(training.device), y.to(training.device)
    #         yhat = training.clf(x)
    #         loss = training.loss_fn(yhat, y)

    #         #Apply backprop
    #         training.opt.zero_grad()
    #         loss.backward()
    #         training.opt.step()

    #     # Print training loss for each epoch
    #     print(f"Epoch: {epoch} loss is {loss.item()}")

    # with open('model_state.pt', 'wb') as f:
    #     save(training.clf.state_dict(), f) 

# Validation loop
    # model.eval()
    # correct = 0
    # total = 0

    # with torch.no_grad():
    #     for images, labels in training.val_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         predicted = (outputs > 0.5).squeeze().long()  # Convert logits to binary predictions (0 or 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # Print validation accuracy
    # print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Testing The Model
    with open('model_state.pt', 'rb') as f:
        training.clf.load_state_dict(load(f))

    image = Image.open('img_2.png')
    image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size (e.g., 224x224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

    image = transform(image)

    img_tensor = image.unsqueeze(0).to(training.device)

    result = torch.argmax(training.clf(img_tensor))

    if result.item() == 1:
        print("It's a G-Man!")
    else:
        print("It's a Bootlegger!")

    

    print(result)
