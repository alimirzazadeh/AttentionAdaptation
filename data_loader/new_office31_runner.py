# /webcam, dslr, amazon


import torchvision.datasets as datasets
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from ipdb import set_trace as bp
import torch.utils.data as data
from PIL import Image
import os.path
import torch.nn.functional as F
import random

def loadOffice31Data(batch_size=4, unsup_batch_size=12, fullyBalanced=True, useNewUnsupervised=True, unsupDatasetSize=None):
    resNetOrInception = 0
    if resNetOrInception == 0:
        imgSize = 256
    else:
        imgSize = 299

    transformations = transforms.Compose([
        #transforms.Resize((299,299)),
        #transforms.RandomRotation(10),
        #transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomVerticalFlip(0.5),
	transforms.RandomResizedCrop(imgSize),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.4,.4,.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    transformations_valid = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    datapath = "/scratch/groups/rubin/alimirz1/office_31/"
    dataset_train_orig = datasets.ImageFolder(datapath + 'dslr/images', transform=transformations)
    dataset_test = datasets.ImageFolder(datapath + 'amazon/images',transform=transformations_valid)


    #random_split(dataset_train_orig, [3, 7], generator=torch.Generator().manual_seed(42))
    train_size = int(0.8 * len(dataset_train_orig))
    valid_size = len(dataset_train_orig) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset_train_orig, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    # unsup_loader = DataLoader(
    #     unsup_train, batch_size=unsup_batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(
        dataset_test, batch_size=1, num_workers=4)


    return train_loader, valid_loader, test_loader


def balancedMiniDataset(trainset, size, limit, fullyBalanced=True):
    counter = np.zeros(10)
    iterating = True
    step = 500
    subsetToInclude = []
    subsetToNotInclude = []
    #subsetToNotInclude += list(range(step))
    wholeRange = list(range(limit))

    random.Random(1234).shuffle(wholeRange)
    subsetToNotInclude = wholeRange[:500]


    while iterating:
        label = trainset[wholeRange[step]][1]
        
        if counter[label] < size:
            counter[label] += 1
            print(counter, step)
            subsetToInclude.append(step)
        # else:
        #     subsetToNotInclude.append(step)
        if np.min(counter) >= size:
            print("Completely Balanced Dataset")
            iterating = False
        if step%1000 == 0:
            print(step)
        step += 1
    # subsetToNotInclude += list(range(step, len(trainset)))
    
    
    #np.savetxt('/home/users/alimirz1/SemisupervisedAttention/saved_batches/2imgclassloosebalanced.out', np.array(subsetToInclude), delimiter=',')
    return torch.utils.data.Subset(trainset, subsetToInclude), torch.utils.data.Subset(trainset, subsetToNotInclude) 
