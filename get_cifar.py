import numpy as np
from itertools import product
import matplotlib.pyplot as plt

########################
### Helper Functions ###
########################

def get(byte):
    return int.from_bytes(byte, byteorder='big')


def show_cifar_10_image(l, img):
    print('Image label:', l)
    img = np.transpose(np.reshape(img, [3, 32, 32]), [1,2,0])
    plt.imshow(img, cmap='hot')
    plt.show()


def show_cifar_100_image(cl, fl, img):
    print('Image label:', cl, '|', fl)
    img = np.transpose(np.reshape(img, [3, 32, 32]), [1,2,0])
    plt.imshow(img, cmap='hot')
    plt.show()


##########################
### CIFAR-10 Functions ###
##########################

def cifar_10_image(index, batch=1, test=False):
    # batch can be 1 through 5
    # index can be 0 through 9,999, or 0 through 999 for testing
    if test:
        path = './cifar-10-batches-bin/test_batch.bin'
        if index > 999:
            raise Exception('CIFAR-10 test index is too large: ' + str(index))
    else:
        path = './cifar-10-batches-bin/data_batch_' + str(batch) + '.bin'
        if index > 9999:
            raise Exception('CIFAR-10 train index is too large: ' + str(index))
        if batch < 1 or batch > 5:
            raise Exception('CIFAR-10 batch number is invalid: ' + str(batch))

    with open(path, 'rb') as f:
        f.seek(index*3073)
        label = get(f.read(1))

        image = np.zeros([3*1024], dtype=np.int16)
        for c, pix in product(range(3), range(1024)):
            image[c*1024+pix] = get(f.read(1))

        return label, image


def cifar_10_labels():
    path = './cifar-10-batches-bin/batches.meta.txt'
    with open(path, 'r') as f:
        labels = []
        for line in f:
            labels.append(line[0:-1])
        return np.array(labels)


###########################
### CIFAR-100 Functions ###
###########################


def cifar_100_image(index, test=False):
    # index can be 0 through 49,999, or 0 through 9,999 for testing
    if test:
        path = './cifar-100-binary/test.bin'
        if index > 9999:
            raise Exception('CIFAR-10 test index is too large: ' + str(index))
    else:
        path = './cifar-100-binary/train.bin'
        if index > 49999:
            raise Exception('CIFAR-10 train index is too large: ' + str(index))

    with open(path, 'rb') as f:
        f.seek(index*3073)
        c_label = get(f.read(1))
        f_label = get(f.read(1))

        image = np.zeros([3*1024], dtype=np.int16)
        for c, pix in product(range(3), range(1024)):
            image[c*1024+pix] = get(f.read(1))

        return c_label, f_label, image


def cifar_100_labels():
    path = './cifar-100-binary/coarse_label_names.txt'
    with open(path, 'r') as f:
        c_labels = []
        for line in f:
            c_labels.append(line[0:-1])

    path = './cifar-100-binary/fine_label_names.txt'
    with open(path, 'r') as f:
        f_labels = []
        for line in f:
            f_labels.append(line[0:-1])

    return np.array(c_labels), np.array(f_labels)
