# Section 3.C Language-Guided Query Selection
# Please download the LSVRC-2012 dataset and set up the MobileClip from https://github.com/apple/ml-mobileclip
# Then modify all the paths to correct ones in your work station

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import mobileclip

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

testset = datasets.CIFAR10(root='./cifar10/', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

device = 'cuda:0'
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')
text_encoder = torch.load('./MobileClip/encoders/text_encoder.pth').eval().to(device)

# collect semantic features of CIFAR-10 classes
text_inputs = torch.cat([tokenizer(f"{c}") for c in testset.classes]).to(device)
text_features = text_encoder(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

# collect semantic features of LSVRC-2012 classes
lsvrc_classes = pd.read_csv('./Class_names/words_all.csv', header=None)

img_features = []
for i in range(100):
    imagenet_inputs = torch.cat([tokenizer(f"{c}") for c in lsvrc_classes[1][i*10:(i+1)*10]]).to(device)
    imagenet_features = text_encoder(imagenet_inputs)
    img_features.append(imagenet_features.detach())

LSVRC_features = torch.cat(img_features, dim=0)
LSVRC_features /= LSVRC_features.norm(dim=-1, keepdim=True)

# calculate the normalized similarity between LSVRC-2012 classes and the overall CIFAR-10 distribution
cosine_similarity = torch.matmul(LSVRC_features, text_features.T)

similarity = torch.sum(cosine_similarity, dim=1)
min = torch.min(similarity)
max = torch.max(similarity)
similarity = (similarity - min) / (max - min)

# calculate the normalized sampling probabilities for each LSVRC-2012 classes
k = 1000

sorted_values, sorted_indices = torch.sort(similarity, descending=True)
top_indices = np.array(sorted_indices[:k].cpu().detach())
top_values = np.array(sorted_values[:k].cpu().detach())

value_index = {}
for index, value in zip(top_indices, top_values):
    if index in value_index:
        value_index[index] += value
        if value_index[index] > 1:
             value_index[index] = 1
    else:
        value_index[index] = value

samples = pd.read_csv("./Class_names/sorted_static.csv", index_col= None)

total_num = 0
for index in value_index:
    total_num += value_index[index] * samples.iloc[index]['File Count']
total_num

# sample from the LSVRC-2012 dataset to construct the query set
import os
import random
import shutil
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

destination_folder = './transfer_set/cifar10/resized_queries/all'

if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder)

N = 100000
multiple = total_num / N
resize_size = (32, 32) 
max_workers = 16 

def process_image(file_name, source_folder, destination_folder, resize_size):
    source_path = os.path.join(source_folder, file_name)
    destination_path = os.path.join(destination_folder, file_name)

    try:
        with Image.open(source_path) as img:
            resized_img = img.resize(resize_size, Image.LANCZOS)
            resized_img.save(destination_path, format='PNG')
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

for index in value_index:
    source_folder = './Imagenet/OpenDataLab___ImageNet-1K/raw/ImageNet-1K/train/' + samples.iloc[index]['Folder Name']
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', 'JPEG'))]
    
    sample_num = round(samples.iloc[index]['File Count'] * value_index[index] / multiple)
    selected_files = random.sample(image_files, sample_num)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_image, file_name, source_folder, destination_folder, resize_size) for file_name in selected_files]

    for future in futures:
        future.result()