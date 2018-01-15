# This is probably the ugliest plotting script ye shall ever see.
# Note to self: kill it with fire after the project is done.

import matplotlib.pyplot as plt
import matplotlib.colors as cls
import matplotlib.patches as mpatches

colors = ['b', 'g', 'r', 'c', 'm']
for nr in range(1,6):
    print "nr = " + str(nr)
    name = "nr-" + str(nr) + "/test.txt"
    open("test.txt", "r")
    i = 0
    if nr ==3:
        name = "test.txt"
        i = 1
    f = open(name, "r")
    count = 0
    a = []
    lines = f.readlines()
    while count < 50:
        line = lines[i]
        if count == 50: break
        a += [str(line.split()[-1].split("%")[0].split(',')[-1])]
        count += 1
        i += 1
    assert len(a) == 50
    f.close()
    plt.plot(range(50), a, c=colors[nr-1])

plt.title("CapsNet testing curves on MNIST for different num_routing_iterations")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(handles=[mpatches.Patch(color=colors[i], label='nri = ' + str(i+1)) for i in range(5)])
plt.savefig('testcurves_fordiff_nri_mnist.png')
