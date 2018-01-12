# Capsule Network Experiments

Started with this code: https://github.com/gram-ai/capsule-networks

So far:
 - factored into 3 files
 - logging to text files
 - argument parser
 - works with cifar10
 - can load/save models
 
Todo:

We need to do something interesting for our project. Some ideas are:

1. Test Hinton's equivariance claim by using a dataset of many 2D projections of a 3D object
 - lead 1: 3d Pascal http://cvgl.stanford.edu/projects/pascal3d.html)
 - roadblock 1: we need an API to take a 3D object and get a 2D projection
   
2. Improve training
 - lead 1: whatever Prem is up to
 - lead 2: increase batch size as form of annealing https://arxiv.org/abs/1711.00489 
 - lead 3: left/right loss terms https://arxiv.org/pdf/1710.09829.pdf

Proposal Feedback:

1. Try to get SOTA on CIFAR

 - "This is a solid goal. Assuming you don't manage to get SOTA within a reasonable amount of time, I would concentrate on analysis of what kind of classes/images it work/fails on and how this relates to ConvNet performance? Is it better at detecting some rotated objects -- try some explicit data augmentation (affine warping) to test this aspect."

 - "Looks great. Will be very interested in seeing the results. CIFAR is exactly the right place to go next."

2. See how CapsNets do on segmentation

 - "Working on real-world datasets might be too ambitious: unless you get very lucky, it won't work easily without a lot of tuning, which you might not have time for - and then it will be hard to draw any conclusions.  Maybe try it, but give up quickly if it doesn't work. Instead, I would suggest creating additional synthetic datasets that test specific hypothesis about what it can or can't do. If you can find tasks that CapsNet are much better or worse at then ConvNet, that would be an interesting result."

 - "For segmentation, consider something like Weizmann horses (http://www.msri.org/people/members/eranb/) rather the PASCAL."

3. Modify CapsNet architecture, components, training procedure

 - "Are you planning to play with the original MNIST data for this? Or will this be based on problems you see in 1,2?"

General.

 - "Additionally, and especially as part of your fall back plan, consider starting from what works (e.g. the MNIST-based data) and then making the task incrementally more difficult until the CapsNet breaks. Map out as many such faliure modes as you can."

 - "Accuracy is less important than insights gained (what variations/hyperparameters did you try? What did you implement?"
