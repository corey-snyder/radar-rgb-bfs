import argparse
from os import listdir
from os.path import isfile, join
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", help="visible cuda device", type=str)
    parser.add_argument("-dir", help="path of new yaml files", type=str)
    args = parser.parse_args()
    device = args.device
    yaml_dir = args.dir

    yamls = listdir(yaml_dir)
    #yamls.sort(key=float)

    for setup_file in yamls:
        os.system("CUDA_VISIBLE_DEVICES=" + str(device) + " python train_radar.py -yaml " + yaml_dir + setup_file)


