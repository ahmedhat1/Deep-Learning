import numpy as np
import matplotlib.pyplot as plt

# create a 8x8 array with alternating ones and zeros
checkerboard = np.array([[0, 1] * 4, [1, 0] * 4] *4)

# use np.tile to repeat the array horizontally and vertically to get a 256x256 checkerboard
#checkerboard = np.tile(checkerboard, (32, 32))

# show the checkerboard using matplotlib
print(checkerboard)
plt.imshow(checkerboard, cmap='gray')
plt.show()
