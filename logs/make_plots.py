# This is probably the ugliest plotting script ye shall ever see.
# Note to self: kill it with fire after the project is done.

import matplotlib.pyplot as plt
import matplotlib.colors as cls
import matplotlib.patches as mpatches
import matplotlib
# effect of nri

j = 1
namemap = {'mnist':"MNIST", 'fashion':"F-MNIST", 'cifar10':"CIFAR10", 'svhn':"SVHN"}
for dset in ['mnist', 'fashion', 'cifar10', 'svhn']:
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 4)
    print dset, 
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    for nr in range(1,7):
        name = dset+"/nr-"+str(nr)+"/test.txt"
        f = open(name, "r")
        a = []
        lines = f.readlines()
        print len(lines)
        for i in range(50):
            line = lines[i]
            a += [str(line.split()[-1].split("%")[0].split(',')[-1])]
        f.close()

        plt.plot(range(50), a, c=colors[nr-1])

    plt.title(namemap[dset])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(handles=[mpatches.Patch(\
                            color=colors[i], label='nri = ' + str(i+1)) for i in range(6)], \
               loc=4)
    fig.savefig('effect_of_nri-'+dset+'.png')
    fig.clf()
    j += 1


namemap = {'mnist':"MNIST", 'fashion':"F-MNIST", 'cifar10':"CIFAR10", 'svhn':"SVHN"}
for dset in ['mnist', 'fashion', 'cifar10', 'svhn']:
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 4)
    print dset
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    runnames = ['/nr-3/train.txt', '/nr-3/test.txt', '/cnn/train.txt', '/cnn/test.txt']
    legendnames = ['CapsNet Train', 'CapsNet Test', 'AlexNet Train', 'AlexNet Test']
    for ind in range(4):
        f = open(dset + runnames[ind], "r")
        a = []
        lines = f.readlines()
        print len(lines),
        for i in range(50):
            line = lines[i]
            a += [str(line.split()[-1].split("%")[0].split(',')[-1])]
        f.close()
        plt.plot(range(50), a, c=colors[ind])

    plt.title(namemap[dset])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(handles=[mpatches.Patch(color=colors[i], label=legendnames[i]) for i in range(4)], loc=4)
    fig.savefig('train-test-'+dset+'.png')
    fig.clf()

        