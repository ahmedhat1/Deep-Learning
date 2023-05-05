import matplotlib.pyplot as plt
from pattern import Checker, Circle, Spectrum


# Create an instance of Checker with resolution=256 and tile_size=32
checker = Checker(250, 25)

# Generate the checkerboard pattern
checker.draw()

# Display the checkerboard pattern
checker.show()

circle = Circle(1024, 200, (512, 256))
circle.draw()
circle.show()

spectrum = Spectrum(255)
spectrum.draw()
spectrum.show()
