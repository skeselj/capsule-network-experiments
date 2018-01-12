"""
Main
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torch.autograd import Variable
import torchnet as tnt
from torchvision.utils import make_grid
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
parser.add_argument("--model_dir", default="epochs", type=str)
parser.add_argument("--starting_epoch", default=0, type=int)
parser.add_argument("--tracking_enabled", default=0, type=int)
parser.add_argument("--max_epochs", default=50, type=int)
parser.add_argument("--num_classes", default=10, type=int)
args = parser.parse_args()

# figure out names and if we're staring fresh
name = "bi-" + str(args.batch_size_init) + "_" + \
       "bg-" + "{0:.4f}".format(args.batch_size_growth) + "_" + \
       "nr-"+ str(args.num_routing_iterations)
model_path = args.model_dir + "/" + args.dataset + "/" + name
log_path = args.log_dir + "/" + args.dataset + "/" + name
starting_fresh = not os.path.exists(model_path +'/epoch_%d.pt' % args.starting_epoch)

# setup dirs if not created
if args.model_dir != '':  
    if not os.path.exists(model_path):
        os.makedirs(model_path)   
if args.log_dir != '' and starting_fresh: 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    f = open(log_path + '/train.txt','w')
    f.close()
    f = open(log_path + '/test.txt','w')
    f.close()

# print info about this run to visdom
if not starting_fresh and args.tracking_enabled:
    text_logger = VisdomTextLogger()
    text_logger.log("dataset: " + "________________ " + str(args.dataset) + " " \
                    "batch_init: "  + "________________ " + str(args.batch_size_init) + " "\
                    "batch_growth: " + "_______________ " + str(args.batch_size_growth) + " "\
                    "num_routing_iterations " + "________ " + str(args.num_routing_iterations) + " "\
                    "starting_epcoh " + "_____________ " + str(args.starting_epoch))

# gpu and dataset
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
if not starting_fresh:
    print("Loading " + model_path)
    model.load_state_dict(torch.load(model_path +'/epoch_%d.pt' % args.starting_epoch))
model.cuda()
capsule_loss = CapsuleLoss()
optimizer = Adam(model.parameters())

### wrappers for metrics (loss, accuracy, confusion)
meter_loss = tnt.meter.AverageValueMeter()
meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
confusion_meter = tnt.meter.ConfusionMeter(args.num_classes, normalized=True)
def reset_meters():
    meter_accuracy.reset()
    meter_loss.reset()
    confusion_meter.reset()
    
### show logs in visdom and log them if tracking_enabled
train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                 'columnnames': list(range(args.num_classes)),
                                                 'rownames': list(range(args.num_classes))})
ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction\n'})


def on_start(state):
     state['epoch'] = args.starting_epoch + 1

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
        if args.dataset == 'mnist':
            ground_truth = test_sample[0].unsqueeze(1).float() / 255.0
        elif args.dataset == 'cifar10':
            ground_truth = test_sample[0].permute(0, 3, 1, 2).float() / 255.0
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data
        ground_truth_logger.log(make_grid(ground_truth, nrow=int(args.batch_size_init ** 0.5),
                                          normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(make_grid(reconstruction, nrow=int(args.batch_size_init ** 0.5),
                                            normalize=True, range=(0, 1)).numpy())
    if args.log_dir != '':
        f = open(log_path + '/test.txt','a')
        f.write(msg + "\n")
        f.close()

    
    torch.save(model.state_dict(), model_path +'/epoch_%d.pt' % state['epoch'])


### engine ties everything together

engine = Engine()
engine.hooks['on_start'] = on_start
engine.hooks['on_sample'] = on_sample
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_forward'] = on_forward
engine.hooks['on_end_epoch'] = on_end_epoch

from utils import augmentation, get_iterator

def processor(sample):
    data, labels, training = sample
    if args.dataset == 'mnist':
        data = data.unsqueeze(1)
    elif args.dataset == 'cifar10':
        data = data.permute(0, 3, 1, 2)
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

engine.train(processor, get_iterator(args.dataset, True), \
             maxepoch=args.max_epochs, optimizer=optimizer)

