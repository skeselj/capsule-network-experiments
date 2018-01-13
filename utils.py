"""
Utils
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

import torchnet as tnt


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(\
                transposed_input.contiguous().view(-1, transposed_input.size(-1)), \
                dim=1)
    return softmaxed_output.view(\
            *transposed_input.size()).transpose(dim, len(input.size()) - 1)

    
def augmentation(x, max_shift=2):
    _, _, height, width = x.size()
    
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
    
    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = \
                x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()
    
                                              
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms

def get_iterator(dataset_name, mode, batch_size=100):
    if dataset_name == "mnist":
        dataset = MNIST(root='./data/mnist', download=True, train=mode),
    elif dataset_name == "fashion":
        dataset = FashionMNIST(root='./data/fashion', download=True, train=mode)
    elif dataset_name == "cifar10":
        #transform = transforms.Compose([transforms.ToTensor(),\
        #                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10(root="./data/cifar10",download=True,train=mode)#, transform=transform)
    data = getattr(dataset, 'train_data' if mode else 'test_data')
    labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
    tensor_dataset = tnt.dataset.TensorDataset([data, labels])
    
    return tensor_dataset.parallel(batch_size=batch_size, num_workers=4, shuffle=mode)
