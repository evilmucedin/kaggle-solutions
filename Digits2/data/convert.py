#!/usr/bin/env python3

import os
import struct
import random
import argparse
from array import array

class MNIST(object):
    def __init__(self, path='.'):
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None, type=int,
                        help="ID (position) of the letter to show")
    parser.add_argument("--binary", default=False,
                        help="produce binary features")
    parser.add_argument("--data", default="./data",
                        help="Path to MNIST data dir")

    args = parser.parse_args()

    mn = MNIST(args.data)

    n = 28
    print("out")
    fOut = open("trainAll.csv", "w")
    print("label", file=fOut, end="")
    for i in range(n*n):
        print(",pixel%d" % i, file=fOut, end="")
    print("", file=fOut)
    
    def out(imgs, labels):
        for img, label in zip(imgs, labels):
            print(label, file=fOut, end="")
            for j in range(n*n):
                if args.binary:
                    feature = 1 if img[j] != 0 else 0
                else:
                    feature = img[j]
                print(",%d" % feature, file=fOut, end="")
            print("", file=fOut)
    
    img, label = mn.load_training()
    out(img, label)
    # img, label = mn.load_testing()
    # out(img, label)

    if args.id:
        which = args.id
    else:
        which = random.randrange(0, len(label))

    print('Showing num: {}'.format(label[which]))
    print(mn.display(img[which]))


