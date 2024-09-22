import torch
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split

testset = datasets.CIFAR10(root='./cifar10/', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

device = 'cuda:0'
student = torch.load('./surrogates/surrogate.pth').eval().to(device)

test_num = len(testloader.dataset)
acc = 0.0
for test_data in testloader:
    test_images, test_labels = test_data
    outputs = student(test_images.to(device))
    predict_y = torch.max(outputs, dim=1)[1]
    acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

test_acc = acc / test_num
print('before alignment:', test_acc)

import torch.nn as nn

class AlgorithmWrapper(nn.Module):
    def __init__(self, student):
        super(AlgorithmWrapper, self).__init__()
        self.featurizer = nn.Sequential(*list(student.children())[:-1])
        self.classifier = student.fc

    def forward(self, x):
        features = self.featurizer(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

algorithm = AlgorithmWrapper(student).to(device)

from T3A import T3A

hparams = {
    'alpha': 0.1,
    'filter_K': 100,
}

t3a_algorithm = T3A(
    input_shape=(3, 32, 32), 
    num_classes=10,          
    num_domains=1,          
    hparams=hparams,        
    algorithm=algorithm 
)

t3a_algorithm.eval()
all_predictions = []
all_labels = []

acc = 0.0
batch = 0
interpolation = False
with torch.no_grad():
    for data, labels in testloader:
        data = data.to(device)
        labels = labels.to(device)

        if batch >= 5:
            interpolation = True
        outputs = t3a_algorithm.predict(data, adapt=True, interpolation=interpolation)
        
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, labels.to(device)).sum().item()
        batch += 1

test_acc = acc / test_num
print('after alignment:', test_acc)