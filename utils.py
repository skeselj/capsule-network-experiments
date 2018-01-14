"""
Utils
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchnet as tnt


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(\
                transposed_input.contiguous().view(-1, transposed_input.size(-1)), \
                dim=1)
    return softmaxed_output.view(\
            *transposed_input.size()).transpose(dim, len(input.size()) - 1)

    
def augmentation(x, max_shift=2):  # translation by up to 2px in each direction
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
    
                                              
import torch.utils.data as data
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
import math

def torchify(x, verbose=False, index=None, top_call=True):
    if type(x) is np.ndarray:
        if verbose and (index == 0 or index==None):
            print("numpy array of shape {}".format(x.shape))
        ret = torch.from_numpy(x)
    elif type(x) is list:
        if verbose and (index == 0 or index==None):
            print("list (length {}) of".format(len(x)))
        ret = torch.cat([torchify(y, index=i, top_call=False).unsqueeze(0) for i, y in enumerate(x)])
    else:
        assert "Tensor" in torch.typename(x)
        if verbose and (index == 0 or index==None):
            print("{} of size {}".format(torch.typename(x), x.size()))
        ret = x
    if top_call:
        print("Final result: {} of size {}".format(torch.typename(ret), ret.size()))
    return ret

class AffinelyTransformed(data.Dataset):
    def __init__(self, dataset, seed=17724):
        self.dataset = dataset
        np.random.seed(seed)
    def __getitem__(self, index):
        return getitem(self.dataset, index)
    def __getattr__(self, attr):
        content = getattr(self.dataset, attr)
        if "data" in attr and "labels" not in attr: # data, not labels
            content = self.affinely_transformed(torchify(content))
        return content
    def __len__(self):
        return len(self.dataset)
    def affinely_transformed(self, content):
        n_samples, img_w = content.size(0), content.size(2) # 2 guaranteed to not be channel
        angle_max, shear_max, disp_max, scale = 20/360, 0.1, 1, 1.5
        angles = np.random.uniform(-angle_max, angle_max, n_samples)
        shears = np.random.uniform(-shear_max, shear_max, n_samples)
        disps = np.random.uniform(-disp_max, disp_max, (n_samples, 2))
        scales = np.repeat([scale], (n_samples))
        thetas = torch.FloatTensor(
            [[[(1+math.cos(angles[i]))/scales[i], math.sin(angles[i]),               disps[i,0]], \
              [shears[i]-math.sin(angles[i]),     (1+math.cos(angles[i]))/scales[i], disps[i,1]]] \
             for i in range(n_samples)])
        grids = F.affine_grid(thetas, size=torch.Size((n_samples,1,img_w,img_w)))

        content = content.float()
        def affine(layer):
            assert layer.size()[1] == layer.size()[2]
            layer = F.grid_sample(layer.unsqueeze(1), grids)
            layer = layer.squeeze().data
            return layer
        if len(content.size()) == 3:
            return affine(content)
        else:
            assert len(content.size()) == 4
            if content.size(1) == 3:
                for dim in range(3):
                    content[:,dim,:,:] = affine(content[:,dim,:,:])
            elif content.size(3) == 3:
                for dim in range(3):
                    content[:,:,:,dim] = affine(content[:,:,:,dim])
            else:
                raise ValueError
            return content



def get_iterator(dataset_name, mode, batch_size=100, trans=False, test=False):
    if dataset_name == "mnist":
        dataset = MNIST(root='./data/mnist', download=True, train=mode)
    elif dataset_name == "fashion":
        dataset = FashionMNIST(root='./data/fashion', download=True, train=mode)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root="./data/cifar10", download=True, train=mode)
    elif dataset_name == "svhn":
        dataset = SVHN(root="./data/svhn", download=True, split=("train" if mode else "test"))
    else:
        raise ValueError

    if trans:
        dataset = AffinelyTransformed(dataset)
    
    if dataset_name in ["mnist", "fashion", "cifar10"]:
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
    elif dataset_name == "svhn":
        data = dataset.data
        labels = dataset.labels
    else:
        raise ValueError
    if test:
        data = data[:batch_size]
        labels = labels[:batch_size]
    tensor_dataset = tnt.dataset.TensorDataset([data, labels])
    
    return tensor_dataset.parallel(batch_size=batch_size, num_workers=4, shuffle=mode)
