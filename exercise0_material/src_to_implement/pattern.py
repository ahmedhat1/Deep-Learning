import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2*tile_size) != 0:
            raise ValueError("Resolution must be evenly divisible by 2*tile size.")

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution))

    def draw(self):
        checkerboard = np.array([[0, 1] , [1, 0]])
        self.output = np.tile(checkerboard,(self.resolution//(2*self.tile_size), self.resolution//(2*self.tile_size)))

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, size=512, radius=100, center=None, color=(255, 255, 255)):
        self.size = size
        self.radius = radius
        self.color = color
        if center is None:
            self.center = (size//2, size//2)
        else:
            self.center = center
        self.output = np.zeros((size, size, 3), dtype=np.uint8)
        x, y = np.indices((size, size))
        self.output[(x-self.center[0])**2 + (y-self.center[1])**2 <= self.radius**2] = color

    def draw(self):
        pass

    def show(self):
        plt.imshow(self.output)
        plt.show()