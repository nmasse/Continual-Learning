import numpy as np
from itertools import product
import matplotlib.pyplot as plt

########################
### Helper Functions ###
########################

def get(byte):
    return int.from_bytes(byte, byteorder='big')


def show_cifar_10_image(l, img, n=0):
    print('Image label:', l[n])
    img = np.transpose(np.reshape(img[n], [3, 32, 32]), [1,2,0])
    plt.imshow(img, cmap='hot')
    plt.show()


def show_cifar_100_image(cl, fl, img, n=0):
    print('Image label:', cl[n], '|', fl[n])
    img = np.transpose(np.reshape(img[n], [3, 32, 32]), [1,2,0])
    plt.imshow(img, cmap='hot')
    plt.show()


##########################
### CIFAR-10 Functions ###
##########################

def cifar_10_image(target, num_images=1, test=False):
    # target can be 0 through 49,999, or 0 through 999 for testing
    if test:
        if not 0 <= target <= 999:
            raise Exception('CIFAR-10 test index is invalid: ' + str(target))
        path = './cifar-10-batches-bin/test_batch.bin'
        index = target
    else:
        if not 0 <= target <= 49999:
            raise Exception('CIFAR-10 train index/batch is invalid: ' + str(target))
        path = './cifar-10-batches-bin/data_batch_' + str(target//10000+1) + '.bin'
        index = target%10000

    with open(path, 'rb') as f:
        f.seek(index*3073)

        labels = np.zeros([num_images], dtype=np.int16)
        images = np.zeros([num_images, 3*1024], dtype=np.int16)
        for n in range(num_images):
            labels[n] = get(f.read(1))
            for c, pix in product(range(3), range(1024)):
                images[n, c*1024+pix] = get(f.read(1))

        return labels, images

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


def cifar_100_image(index, num_images=1, test=False):
    # index can be 0 through 49,999, or 0 through 9,999 for testing
    if test:
        path = './cifar-100-binary/test.bin'
        if not 0 < index < 9999:
            raise Exception('CIFAR-100 test index is invalid: ' + str(index))
    else:
        path = './cifar-100-binary/train.bin'
        if not 0 < index < 49999:
            raise Exception('CIFAR-100 train index is invalid: ' + str(index))

    with open(path, 'rb') as f:
        f.seek(index*3073)

        c_labels = np.zeros([num_images], dtype=np.int16)
        f_labels = np.zeros([num_images], dtype=np.int16)
        images   = np.zeros([num_images, 3*1024], dtype=np.int16)
        for n in range(num_images):
            c_labels[n] = get(f.read(1))
            f_labels[n] = get(f.read(1))
            for c, pix in product(range(3), range(1024)):
                images[n, c*1024+pix] = get(f.read(1))

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
