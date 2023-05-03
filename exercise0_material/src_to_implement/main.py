import matplotlib.pyplot as plt
from pattern import Checker

# Create an instance of Checker with resolution=256 and tile_size=32
checker = Checker(256, 32)

# Generate the checkerboard pattern
checker.draw()

# Display the checkerboard pattern
checker.show()
plt.show()
