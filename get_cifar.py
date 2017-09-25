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
    # target can be 0 through 49,999, or 0 through 9,999 for testing
    if test:
        if not 0 <= target <= 9999:
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
        if not 0 <= index <= 9999:
            raise Exception('CIFAR-100 test index is invalid: ' + str(index))
    else:
        path = './cifar-100-binary/train.bin'
        if not 0 <= index <= 49999:
            raise Exception('CIFAR-100 train index is invalid: ' + str(index))

    with open(path, 'rb') as f:
        f.seek(index*3074)

        c_labels = np.zeros([num_images], dtype=np.int16)
        f_labels = np.zeros([num_images], dtype=np.int16)
        images   = np.zeros([num_images, 3*1024], dtype=np.int16)
        for n in range(num_images):
            c_labels[n] = get(f.read(1))
            f_labels[n] = get(f.read(1))
            for c, pix in product(range(3), range(1024)):
                images[n, c*1024+pix] = get(f.read(1))

        return c_labels, f_labels, images


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


#########################
### Data Access Class ###
#########################

class DataSet:

    def __init__(self, c_id):
        self.current_index = 0
        self.max_index = 49999
        self.num_examples = 50000
        self.c_id = c_id

        if self.c_id == 'cifar-10':
            self.train_func = cifar_10_image
        elif self.c_id == 'cifar-100':
            self.train_func = cifar_100_image

    def set_index(self, new_index):
        self.current_index = new_index

    def get_image_batch(self, b_size):
        if self.current_index+b_size > self.max_index:
            images_left  = self.max_index-self.current_index
            images_to_go = (self.current_index+b_size)%self.num_examples+1

            *labels1, images1 = self.train_func(self.current_index, num_images=images_left)
            *labels2, images2 = self.train_func(0, num_images=images_to_go)

            labels = np.concatenate([labels1, labels2], axis=1)
            images = np.concatenate([images1, images2])
            self.current_index = images_to_go

        else:
            *labels, images = self.train_func(self.current_index, num_images=b_size)
            self.current_index += b_size

        return images, labels

    def label_conversion(self, labels, b_size):
        if self.c_id == 'cifar-10':
            v_output = np.zeros([b_size, 10])
            for i in range(b_size):
                v_output[i,labels[0][i]] = 1
        elif self.c_id == 'cifar-100':
            raise Exception('CIFAR-100 labels not yet implemented.')

        return v_output

    def next_batch(self, b_size):
        inputs, labels = self.get_image_batch(b_size)
        outputs = self.label_conversion(labels, b_size)
        print(np.shape(inputs), np.shape(outputs))
        quit()

    def get_test_images(self):
        # Currently takes 40 seconds to fully retrieve all images
        *labels, images = self.train_func(0, num_images=10000, test=True)


"""
d = DataSet('cifar-10')
d.set_index(4700)
for i in range(5):
    d.next_batch(13)
    quit()


pass
"""
