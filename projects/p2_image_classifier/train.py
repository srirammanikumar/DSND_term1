import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os

parser = argparse.ArgumentParser(description='Program to train image classifier')

parser.add_argument('--save_dir', default='checkpoint.pth')
parser.add_argument('--data_dir', default='flowers')
parser.add_argument('--arch', help='Select your pre-trained network', choices=['vgg11', 'vgg16', 'vgg19'],
                    default='vgg11')
parser.add_argument('--learning_rate', type=float, help='Chose a learning rate for your network', default=0.0003)
parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=4096)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=3)
parser.add_argument('--gpu', help='Enter either GPU or CPU', default='GPU')

args = parser.parse_args()

# Setting directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

output_size = len(os.listdir(train_dir))

# Set device to cuda or cpu
device = torch.device("cuda" if (args.gpu == "GPU" and torch.cuda.is_available()) else "cpu")

# Create transforms
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# Using the image datasets and the transforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)

# Train model
model = getattr(models, args.arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                           nn.ReLU(),
                           nn.Dropout(0.4),
                           nn.Linear(args.hidden_units, 1024),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(1024, output_size),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)
epochs = args.epochs
steps = 0

print_every = 10
for epoch in range(epochs):
    running_loss = 0
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)

                    equals = (labels.data == ps.max(dim=1)[1])
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Valid loss: {valid_loss / len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy / len(validloader):.3f}")
            running_loss = 0
            model.train()

print("Model training complete...")
# Saving the checkpoint
model.class_to_idx = train_data.class_to_idx

checkpoint = {'input_size': 25088,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'model_classifier': model.classifier,
              'optimizer_dict': optimizer.state_dict(),
              'model_mapping': model.class_to_idx,
              'epochs': epochs,
              'model_name': args.arch
              }

torch.save(checkpoint, args.save_dir)
print("Model saved")