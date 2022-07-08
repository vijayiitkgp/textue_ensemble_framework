import torch
import matplotlib.pyplot as plot
from torchvision import datasets, transforms

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.3,)),
                                 ])

train_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=60, shuffle=True)
from torch.utils.data import Dataset
import random


class SampleDataset(Dataset):
    def __init__(self, r, r1):
        random_list = []
        for x in range(2, 999):
            m = random.randint(r, r1)
            random_list.append(m)
        self.samples = random_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx])


datasets = SampleDataset(2, 440)
datasets[90:100]
from torch.utils.data import DataLoader

dloader = DataLoader(datasets, batch_size=10, shuffle=True, num_workers=4)
for x, batch in enumerate(dloader):
    print(x, batch)