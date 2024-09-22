import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader, Subset, random_split

transform_train = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

crop_24 = transforms.RandomResizedCrop(size=(24, 24), scale=(0.5, 1.0))

trainset = datasets.ImageFolder(root='./transfer_set/cifar10/resized_queries/', transform=transform_train)
testset = datasets.CIFAR10(root='./cifar10/', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)

def extract_loop(model, teacher, poi_loader, loader, opt, lr_scheduler, epoch,
                temperature=100.0, max_epoch=100, mode='train', device='cuda'):

    T = temperature
    
    if mode != 'train':
        model.eval()
        test_num = len(loader.dataset)
        acc = 0.0
        for test_data in loader:
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

        test_accurate = acc / test_num
        print('test acc:', test_accurate)
        
        return None

    for batch_idx, batch in enumerate(loader):
        if mode == 'train':
            model.train()
        else:
            model.eval()
        images = batch[0]
        labels = batch[1].long()

        images = images.to(device)
        labels = labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()

        # Section 3.A Two-Stage Extraction with Varying Resolution
        if epoch < 20:
            preds = model(crop_24(images))
        else:
            preds = model(images)
        teacher_preds = teacher(images)
        
        # Section 3.B Temperature Scaling in Black-box Extraction
        # Direct distillation with teacher logits: F.softmax(teacher_preds / T, dim=-1)
        # Temperature scaling in black-box extraction: F.softmax(torch.log(F.softmax(teacher_preds, dim=-1)) / T, dim=-1)
        # The output from black-box victim API is F.softmax(teacher_preds, dim=-1),
        # and F.softmax(torch.log(F.softmax(teacher_preds, dim=-1)) / T, dim=-1) derives F.softmax(teacher_preds / T, dim=-1) from F.softmax(teacher_preds, dim=-1).
        # To mitigate numerical errors, a small ε could be added to the victim output pseudo-logits before applying the log operation. However, in our experiments, this adjustment appears unnecessary.
        extract_loss = T ** 2 * F.kl_div(F.log_softmax(preds / T, dim=-1), F.softmax(torch.log(F.softmax(teacher_preds, dim=-1)) / T, dim=-1), reduction='batchmean')

        if mode == 'train':
            extract_loss.backward()
            opt.step()

device = 'cuda:0'

def extraction(teacher, model, epochs, poi_loader, train_loader, test_loader, opt, lr_scheduler, device):

    teacher.eval()
    print('Teacher accuracy:')
    test_num = len(test_loader.dataset)
    acc = 0.0
    for test_data in test_loader:
        test_images, test_labels = test_data
        outputs = teacher(test_images.to(device))
        predict_y = torch.max(outputs, dim=1)[1]
        acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

    test_acc = acc / test_num
    print(test_acc)
    
    for epoch in range(epochs):
        print('epoch:', epoch)
        model.train()
        extract_loop(model, teacher, poi_loader, train_loader,
                opt, lr_scheduler, epoch, max_epoch=epochs, mode='train', device=device)

        with torch.no_grad():
            model.eval()
            extract_loop(model, teacher, poi_loader, test_loader,
                opt, lr_scheduler, epoch, max_epoch=epochs, mode='val', device=device)
        model.eval()
        
        lr_scheduler.step()

teacher = torch.load('./victims/resnet18_50epochs_0.pth').to(device)
from torchvision.models.resnet import resnet18, ResNet18_Weights
student = resnet18(weights=ResNet18_Weights.DEFAULT)
student.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
student.maxpool = nn.Identity()
student.fc = nn.Linear(512,10)
student.to(device)
print('surrogate prepared.')

import math

class CustomLR:
    def __init__(self, optimizer, eta_max, eta_min, T_max1, T_max2, eta_rise):
        self.optimizer = optimizer
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        self.eta_rise = eta_rise
        self.epoch = 0

    def step(self):
        if self.epoch <= self.T_max1:
            lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(self.epoch / self.T_max1 * math.pi))
        elif self.epoch <= self.T_max1 + self.T_max2:
            adjusted_epoch = self.epoch - self.T_max1
            lr = self.eta_rise + 0.5 * (self.eta_rise - self.eta_min) * (1 + math.cos(adjusted_epoch / self.T_max2 * math.pi))
        else:
            lr = self.eta_min
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.epoch += 1

lr = 5e-3
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(student.parameters(), lr=lr, weight_decay=0.0, momentum=0.9)
scheduler = CustomLR(optimizer=optimizer, eta_max=5e-3, eta_min=5e-4, T_max1=20, T_max2=5, eta_rise=1e-3)

extraction(teacher=teacher, model=student, epochs=25, poi_loader=None, train_loader=trainloader, test_loader=testloader, opt=optimizer, lr_scheduler=scheduler, device=device)

import os
import shutil

if not os.path.exists('./surrogates/'):
    os.makedirs('./surrogates/')
    
torch.save(student, './surrogates/surrogate.pth')