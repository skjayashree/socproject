import numpy as np
from PIL import Image

# Make sure that image for you want to get output should be present
# in same folder as of python file. Run this code using some others images too :)
Image_name = "c.png"

# Open the image file
img = Image.open(Image_name)

# Convert the image to a numpy array and extract the grayscale values (ravel() extracts the grayscale values )
img_array = np.array(img)
gray_values = img_array.ravel()
# Now numpy array gray_values contains gray value of each pixel.

##### TO DO #####
# Using Introduction.py see If image contains all pixel values are zero or not.
# Convert the given black image (Image is black because all grayscale values are near to zero) into
# img_stretched so that our eyes can see what's actually there in the image.
# This process is a real life application in Biology using Digital Image Processing.
# Change something in img_array :)
# Run your code with given 3 images by changing Image name in the code.

# HINTS
# 1. Increase the contrast (Read last week theory- Histogram Stretching)
# 2. min_value = np.min(gray_values) - This gives minimum value from the numpy array gray_values. Similarly you can find maximum value :)

# ACTIVITY
# I have given three images a.png, b.png, c.png. The time order of taking this image is as follows:
# a.png b.png c.png
#   t    t+T   t+2T
# Can you guess process?
file = open("output.txt", "w")

# Get size of the image (Image is a matrix) into file
file.write(f"Size of image = {img.size}")

# Loop through each pixel and print its RGB value and gray_scale value
# Task 1: Check if all pixel values are zero
if np.all(gray_values == 0):
    print("All pixel values are zero.")

# Task 2: Stretch the contrast of the black image
min_value = np.min(gray_values)
max_value = np.max(gray_values)

# Apply contrast stretching formula
img_array = (img_array - min_value) * (255 / (max_value - min_value))

# Task 3: Change something in img_array (e.g., invert the image)
img_array = 255 - img_array


#### WRITE YOUR CODE HERE ####

############# END ############


# Convert the numpy array back to an image and save it
img_stretched = Image.fromarray(np.uint8(img_array))
img_stretched.save(f"final-{Image_name}")