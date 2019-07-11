import argparse
import torch
from torchvision import datasets, transforms, models

from PIL import Image
import numpy as np
import json


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    width, height = im.size
    im = im.resize((256, int(256 * (height / width))) if width < height else (int(256 * (width / height)), 256))
    width, height = im.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    im = im.crop((left, top, right, bottom))

    np_image = np.array(im) / 255
    np_image -= np.array([0.485, 0.456, 0.406])
    np_image /= np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, trained_model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    trained_model.to(device)
    # Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze_(0)
    image = image.float()
    image = image.to(device)

    trained_model.eval()
    with torch.no_grad():
        output = trained_model.forward(image)

    ps = torch.exp(output)

    top_p, top_index = ps.topk(topk, dim=1)

    prob_list = np.array(top_p[0])
    index_list = np.array(top_index[0])

    class_to_idx = trained_model.class_to_idx

    index_to_class = {x: y for y, x in class_to_idx.items()}

    class_list = []
    for index in index_list:
        class_list += [index_to_class[index]]

    return prob_list, class_list


parser = argparse.ArgumentParser(description='Program to train image classifier')

parser.add_argument('--image_path', help='Give the path to the input image', required=True)
parser.add_argument('--save_dir', help='Give the path to the checkpoint', required=True)
parser.add_argument('--top_k', help='Top no of predictions', type=int, default=5)
parser.add_argument('--category_names', help='Category to name mapping json', default='cat_to_name.json')
parser.add_argument('--gpu', help='Enter either GPU or CPU', default='GPU')

args = parser.parse_args()

# load the category to class mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Set device to cuda or cpu
device = torch.device("cuda" if (args.gpu == "GPU" and torch.cuda.is_available()) else "cpu")

# Load from checkpoint and create a model
checkpoint = torch.load(args.save_dir)
model = getattr(models, checkpoint['model_name'])(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.class_to_idx = checkpoint['model_mapping']
model.classifier = checkpoint['model_classifier']
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

top_prob, top_class = predict(args.image_path, model, args.top_k)

# To get class names from indexes
species = []
for index in top_class:
    species += [cat_to_name[index]]

print(f"Top probabilities: {top_prob}"
      f"Top classes:: {species}")
