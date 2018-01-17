"""
alex_train.py

Train AlexNet 
"""

import sys
import os
from collections import defaultdict
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torchnet as tnt
from torchvision.utils import make_grid
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
import torchvision.models as models

parser = argparse.ArgumentParser()
# hyperparameters
parser.add_argument("--batch_size", default=100, type=int)
# affect operation of training/testing
parser.add_argument("--dataset", type=str, default="mnist", help="cifar10, svhn")
parser.add_argument("--transform", action="store_true", help="affinely transform data when testing")
parser.add_argument("--max_epochs", default=50, type=int)
# logging
parser.add_argument("--log_dir", default="logs")
parser.add_argument("--tag")
parser.add_argument("-t", "--track", action="store_true")
# system
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument("--visdom_port", default=8097, type=int)
args = parser.parse_args()

####################################################################################################

# figure out names and if we're staring fresh (which means we'll rewrite logs and not load models)
name = "cnn" + ("-trans" if args.transform else "")
if args.tag is not None:
    name += "-" + args.tag
log_path = os.path.join(args.log_dir, args.dataset, name)
visdom_env = args.dataset + "-" + name
if args.track:
    assert args.visdom_port is not None

# setup dirs if not created
if args.log_dir != '':
    os.makedirs(log_path, exist_ok=True)
    f = open(log_path + '/train.txt','w')
    f.close()
    f = open(log_path + '/test.txt','w')
    f.close()

# configure gpu and dataset
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if args.dataset in ['mnist', 'fashion']:
    img_channels = 1
    img_width = 28
elif args.dataset in ['cifar10', 'svhn']:
    img_channels = 3
    img_width = 32
else:
    raise ValueError
def classes(dataset):
    if dataset in ['mnist', 'fashion', 'cifar10', 'svhn']:
        return 10
    else:
        raise ValueError
num_classes = classes(args.dataset)

### specify model and its loss
from cnn_model import smallnet
model = smallnet(in_channels=img_channels)
model.cuda()
capsule_loss = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.0005)

### wrappers for metrics (loss, accuracy, confusion)
meter_loss = tnt.meter.AverageValueMeter()
meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)
def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()
    
### show logs in visdom and log them if track
train_loss_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Train Loss'}, port=args.visdom_port)
train_error_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Train Accuracy'}, port=args.visdom_port)
test_loss_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Test Loss'}, port=args.visdom_port)
test_accuracy_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Test Accuracy'}, port=args.visdom_port)
confusion_logger = VisdomLogger('heatmap', env=visdom_env, opts={'title': 'Confusion matrix',
                                                 'columnnames': list(range(num_classes)),
                                                 'rownames': list(range(num_classes))})

####################################################################################################

def on_sample(state):
    state['sample'].append(state['train'])

def on_start_epoch(state):
    reset_meters()
    if args.track:
        state['iterator'] = tqdm(state['iterator'])

def on_forward(state):
    meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].data[0])
    confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))

def on_end_epoch(state):
    # train
    msg = '[%s] [Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        visdom_env, state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0])
    if args.log_dir != '':
        f = open(log_path + '/train.txt','a')
        f.write(msg + "\n")
        f.close()
    if args.track:
        print(msg)
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])
    reset_meters()
    # test
    engine.test(processor, get_iterator(args.dataset, False, args.batch_size, trans=args.transform))
    msg = '[%s] [Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        visdom_env, state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0])
    if args.track:
        print(msg)
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())
    if args.log_dir != '':
        f = open(log_path + '/test.txt','a')
        f.write(msg + "\n")
        f.close()

### engine ties everything together

engine = Engine()
engine.hooks['on_sample'] = on_sample
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_forward'] = on_forward
engine.hooks['on_end_epoch'] = on_end_epoch

from utils import augmentation, get_iterator

def process(data):
    if args.dataset in ['mnist', 'fashion']:
        data = data.unsqueeze(1)
    elif args.dataset == 'cifar10':
        data = data.permute(0, 3, 1, 2)
    elif args.dataset == 'svhn':
        pass # explicit
    else:
        raise ValueError
    assert torch.max(data) > 2 # To ensure everything needs to be scaled down
    return data.float() / 255.0

def processor(sample):
    data, labels, training = sample
    data = augmentation(process(data))
    labels = torch.LongTensor(labels)
    labels = torch.sparse.torch.eye(num_classes).index_select(dim=0, index=labels)
    data = Variable(data).cuda()
    classes = F.softmax(model(data).cuda(), dim=1)
    labels = Variable(labels, requires_grad=False).cuda()
    loss = capsule_loss(classes, labels)
    return loss, classes

engine.train(processor, get_iterator(args.dataset, True, test=False), \
             maxepoch=args.max_epochs, optimizer=optimizer)
