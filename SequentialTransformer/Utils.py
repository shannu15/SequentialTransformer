import numpy as np
import torch
from torch.utils.data import DataLoader
import gzip
import argparse
import torchvision
import torchvision.transforms as transforms


def get_train_loader(dataset, batch_size):
  data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)
  return data_loader

def get_test_loader(dataset, batch_size):
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
  return data_loader

def split_train_val(train, val_split):
  train_len = int(len(train) * (1.0-val_split))
  train, val = torch.utils.data.random_split(train, (train_len, len(train) - train_len), generator=torch.Generator().manual_seed(42),)
  return train, val

def get_loaders_CIFAR10(seq_len, batch_size, grayscale):
  if grayscale:
    transform = transforms.Compose([
      transforms.Grayscale(),
      transforms.ToTensor(),
      transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
      transforms.Lambda(lambda x: x.view(1, 1024).t())])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      transforms.Lambda(lambda x: x.view(3, 1024).t())])
    # no data augmentation
  transform_train = transform_test = transform
  trainset = torchvision.datasets.CIFAR10(root='./data/cifar/', train=True, download=True, transform=transform_train)
  trainset, _ = split_train_val(trainset, val_split=0.01)
  valset = torchvision.datasets.CIFAR10(root='./data/cifar/', train=True, download=True, transform=transform_test)
  _, valset = split_train_val(valset, val_split=0.01)
  testset = torchvision.datasets.CIFAR10(root='./data/cifar/', train=False, download=True, transform=transform_test)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
  return trainloader, testloader, testloader