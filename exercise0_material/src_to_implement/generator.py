import json
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy import random
from skimage import transform as tf

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next
# function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time
# it gets called.
# This input consists of a batch of images and its corresponding labels.


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.batch_number = 0
        self.current_epoch_number = -1
        with open(self.label_path, "r") as f:
            self.labels = json.load(f)
        self.labels_copy = [int(i[0]) for i in list(self.labels.items())]
        self.batches_per_epoch = (len(self.labels)//self.batch_size + 1) if (
            len(self.labels) % self.batch_size) != 0 else len(self.labels)//self.batch_size
        self.seed = 1234
        self.list_of_batches = []

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # TODO: implement next method
        # for i in range(self.batch_size)
        # list of batches:
        if not self.shuffle:
            np.random.seed(self.seed)
        if len(self.labels_copy) > (len(self.labels) % self.batch_size):
            images_indices = random.choice(
                self.labels_copy, self.batch_size, replace=False)
            self.list_of_batches.append(images_indices)
        else:
            images_indices = random.choice(
                self.labels_copy, len(self.labels_copy), replace=False)
            self.labels_copy = [int(i[0]) for i in list(self.labels.items())]
            remaining = self.list_of_batches[0][:(
                self.batch_size - len(images_indices))]
            images_indices = list(images_indices)
            remaining = list(remaining)
            images_indices.extend(remaining)
            images_indices = np.array(images_indices)
        #print(images_indices)
        images = []
        labels = []
        for image_index in images_indices:
            image = np.load(f"{self.file_path}/{image_index}.npy")
            if image.shape != self.image_size:
                image = tf.resize(image, output_shape=self.image_size)
            image = self.augment(image)
            # self.class_name(image_index)
            label = self.labels.__getitem__(str(image_index))
            images.append(image)
            labels.append(label)
            self.labels_copy.remove(image_index)
        if self.batch_number == 0:
            self.current_epoch_number += 1
        self.batch_number += 1
        if self.batch_number == self.batches_per_epoch:
            self.batch_number = 0
            self.list_of_batches = []
            self.labels_copy = [int(i[0]) for i in list(self.labels.items())]

        # print(labels)
        return np.array(images), labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        # TODO: implement augmentation function
        if self.mirroring:
            np.random.seed(None)
            mirror = random.choice([0, 1], 1)
            if mirror:
                # flipping horizontaly
                img = img[:, ::-1]
        if self.rotation:
            np.random.seed(None)
            rotation = random.choice([0, 90, 180, 270], 1)
            img = tf.rotate(img, rotation[0])
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.current_epoch_number

    def class_name(self, x):
        # This function returns the class name for a specific input
        # TODO: implement class name function
        name = self.class_dict.__getitem__(x)
        return name

    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        # TODO: implement show method
        images, labels = self.next()
        rows = int(np.ceil(np.sqrt(self.batch_size)))
        cols = int(np.ceil(self.batch_size / rows))
        fig, axs = plt.subplots(rows, cols)
        image_number = 0
        for i in range(rows):
            for j in range(cols):
                if image_number < self.batch_size:
                    axs[i, j].imshow(images[image_number])
                    axs[i, j].set_title(self.class_name(labels[image_number]))
                    axs[i, j].axis('off')
                    image_number += 1
                if image_number >= self.batch_size:
                    axs[i, j].axis('off')

        plt.show()

        pass
