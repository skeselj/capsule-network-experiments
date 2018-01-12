"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.

Current modifications:
 - made utils file and model file
 - don't plot confusion matrix or reconstructions
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torch.autograd import Variable
import torchnet as tnt
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
from tqdm import tqdm

import sys
import os
import argparse
parser = argparse.ArgumentParser()
# hyperparameters
parser.add_argument("--batch_size_init", default=100, type=int)
parser.add_argument("--batch_size_growth", default=0, type=float)
parser.add_argument("--num_routing_iterations", default=3, type=int)
# other parameters
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument("--dataset",type=str,default="cifar10")   # mnist, cifar10
parser.add_argument("--log_dir", default="logs", type=str)
parser.add_argument("--tracking_enabled", default=0, type=int)
parser.add_argument("--num_epochs", default=50, type=int)
parser.add_argument("--num_classes", default=10, type=int)
args = parser.parse_args()

text_logger = VisdomTextLogger()
text_logger.log("Info about this run: \n\n" + \
                "dataset: " + str(args.dataset) + "\n\n" + \
                "batch_init" + str(args.batch_size_init) + "\n\n" + \
                "batch_growth: " + "{0:.4f}".format(args.batch_size_growth) + "\n\n" + \
                "num_routing_iterations: " + str(args.num_routing_iterations))

name = "ds-" + str(args.dataset) + "_" + \
       "bi-" + str(args.batch_size_init) + "_" + \
       "bg-" + "{0:.4f}".format(args.batch_size_growth) + "_" + \
       "nr-"+ str(args.num_routing_iterations)
log_path = args.log_dir + "/" + name
if args.log_dir != '':
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path + '/train.txt','w')
    f.close()
    f = open(log_path + '/test.txt','w')
    f.close()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

if args.dataset == 'mnist':
    img_channels = 1
    magic_number = 6
    width = 28
elif args.dataset == 'cifar10':
    img_channels = 3
    magic_number = 8
    width = 32

    
### model and its loss
from model import CapsuleLayer, CapsuleNet, CapsuleLoss
model = CapsuleNet(img_channels, args.num_classes, args.num_routing_iterations, magic_number, width)
model.cuda()
capsule_loss = CapsuleLoss()
optimizer = Adam(model.parameters())


### metric wrappers
meter_loss = tnt.meter.AverageValueMeter()
meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)
def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()
    

### show logs in visdom and log them if log_ir != ''

train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                 'columnnames': list(range(args.num_classes)),
                                                 'rownames': list(range(args.num_classes))})
ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction\n'})


def on_sample(state):
    state['sample'].append(state['train'])
    
def on_start_epoch(state):
    reset_meters()
    if args.tracking_enabled:
        state['iterator'] = tqdm(state['iterator'])

def on_forward(state):
    meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].data[0])
    confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))

def on_end_epoch(state):
    # train
    msg = '[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0])
    if args.log_dir != '':
        f = open(log_path + '/train.txt','a')
        f.write(msg + "\n")
        f.close()
    if args.tracking_enabled:
        print(msg)
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])
    reset_meters()
    # test
    engine.test(processor, get_iterator(args.dataset, False, args.batch_size_init))
    msg = '[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
        state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0])
    if args.tracking_enabled:
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())
        # reconstructions
        test_sample = next(iter(get_iterator(args.dataset,False)))  # False sets value of train mode
        ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5),
                                          normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5),
                                            normalize=True, range=(0, 1)).numpy())
    if args.log_dir != '':
        f = open(log_path + '/test.txt','a')
        f.write(msg + "\n")
        f.close()

    #torch.save(model.state_dict(), 'epochs/'+name+'/epoch_%d.pt' % state['epoch'])


### engine ties everything together

engine = Engine()
engine.hooks['on_sample'] = on_sample
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_forward'] = on_forward
engine.hooks['on_end_epoch'] = on_end_epoch

from utils import augmentation, get_iterator

def processor(sample):
    data, labels, training = sample
    if args.dataset == 'mnist': data = data.unsqueese(1)
    elif args.dataset == 'cifar10': data = data.permute(0, 3, 1, 2)
    data = augmentation(data.float() / 255.0)
    labels = torch.LongTensor(labels)
    labels = torch.sparse.torch.eye(args.num_classes).index_select(dim=0, index=labels)
    data = Variable(data).cuda()
    labels = Variable(labels).cuda()
    if training:
        classes, reconstructions = model(data, labels)
    else:
        classes, reconstructions = model(data)
    loss = capsule_loss(data, labels, classes, reconstructions)
    return loss, classes

#def on_start(state):
#     state['epoch'] = 327
#engine.hooks['on_start'] = on_start

engine.train(processor, get_iterator(args.dataset, True), \
             maxepoch=args.num_epochs, optimizer=optimizer)

