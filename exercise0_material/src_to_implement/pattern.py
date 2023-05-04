import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2*tile_size) != 0:
            raise ValueError(
                "Resolution must be evenly divisible by 2*tile size.")

        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution, resolution))
        # self.zero_tile = np.zeros((self.tile_size, self.tile_size))
        # self.one_tile = np.ones((self.tile_size, self.tile_size))

    def draw(self):

        black_tile = np.zeros((self.tile_size, self.tile_size))  # size (25*25)
        white_tile = np.ones((self.tile_size, self.tile_size))  # size (25*25)
        checkerboard_1 = np.concatenate((black_tile, white_tile), axis=1)
        checkerboard_2 = np.concatenate((white_tile, black_tile), axis=1)
        checkerboard = np.concatenate((checkerboard_1, checkerboard_2), axis=0)
        # checkerboard = np.array(
        #     [[black_tile, white_tile], [white_tile, black_tile]])
        self.output = np.tile(
            checkerboard, (self.resolution//(2*self.tile_size), self.resolution//(2*self.tile_size)))
        self.output = self.output
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution=512, radius=100, center=None):
        self.resolution = resolution
        self.radius = radius
        self.center = center
        self.rad_sq = self.radius ** 2
        if center is None:
            self.center = (resolution//2, resolution//2)
        else:
            self.center = center
        self.output = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        # x, y = np.indices((resolution, resolution))

    def draw(self):
        x = np.arange(0, self.resolution, 1)
        y = np.arange(0, self.resolution, 1)
        xx, yy = np.meshgrid(x, y)
        # xx = xx[:,0]
        # yy = np.transpose(yy)
        # xx = np.transpose(xx)
        # yy = np.flip(yy)
        circle = (xx-self.center[0])**2 + (yy-self.center[1])**2
        # self.output = circle
        self.output = (circle < (
            self.rad_sq - self.center[0])) & (circle < (self.rad_sq - self.center[1]))
        self.output = self.output.astype(np.int64)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
