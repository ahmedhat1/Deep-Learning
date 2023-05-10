import numpy as np
import matplotlib.pyplot as plt

# resolution that defines the number of pixels in each dimension, and an integer tile size that defines the
# number of pixel an individual tile has in each dimension
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

        #1*2 tiles
        checkerboard_1 = np.concatenate((black_tile, white_tile), axis=1)
        checkerboard_2 = np.concatenate((white_tile, black_tile), axis=1)
        #concatenate the two(1*2) rows create a 2*2 checkerboard
        checkerboard = np.concatenate((checkerboard_1, checkerboard_2), axis=0)
        self.output = checkerboard
#       repeat the checkerboard pattern across the entire output grid by tiling the checkerboard r//2*t_s times
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

    def draw(self):
        x = np.arange(0, self.resolution)
        y = np.arange(0, self.resolution)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        circle = (xx-self.center[0])**2 + (yy-self.center[1])**2
        #points within the circle < radius squared r^2 = (x-h)^2 + (y-k)^2
        self.output = (circle < self.rad_sq).astype(bool)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray', origin='lower')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((resolution, resolution, 3))

    def draw(self):
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx,yy = np.meshgrid(x, y)

        spectrum = np.zeros((self.resolution, self.resolution, 3))
        # [:, np.newaxis][::-1]  # red
        spectrum[:, :, 0] = np.linspace(0, 1, self.resolution) #red channel intensity increasing from left to right
        spectrum[:, :, 1] = (np.linspace(0, 1, self.resolution))[:, np.newaxis] #blue channel incresing from top to bottom
        spectrum[:, :, 2] = np.linspace(1, 0, self.resolution) #green channel intensity decreasing from left to right

        self.output = spectrum
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()
