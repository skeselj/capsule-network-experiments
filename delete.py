from itertools import product
import argparse
import os
import shutil

datasets = ["mnist", "cifar10", "fashion", "svhn"]
experiments = ["nr-3", "nr-3-trans"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", action="store_true")
    for dataset, experiment in product(datasets, experiments):
        d = os.path.join("tb", dataset, experiment)
        timestamps = os.listdir(d)
        for timestamp in timestamps:
            subdir = os.path.join(d, timestamp)
            for epoch, it in product(range(1, 50), range(4)):
                embedding = os.path.join(subdir,
                                         str(epoch * 100 + it).zfill(5))
                if os.path.exists(embedding):
                    print(embedding)
                    if not args.test:
                        shutil.rmtree(embedding)
