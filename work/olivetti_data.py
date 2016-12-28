from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt

olivetti = fetch_olivetti_faces()

# set up the figure
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1,\
  hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(8):
  ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
  ax.imshow(olivetti.images[0+i], cmap=plt.cm.bone, \
    interpolation='nearest')

  # label the image with the target value
  ax.text(0, 7, str(olivetti.target[i]))

plt.show()
