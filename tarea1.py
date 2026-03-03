import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("images.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

hist_R = np.bincount(R.ravel(), minlength=256)
hist_G = np.bincount(G.ravel(), minlength=256)
hist_B = np.bincount(B.ravel(), minlength=256)

x = np.arange(256)

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("imagen")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.plot(x, hist_R)
plt.title("histograma R")
plt.xlim(0, 255)
plt.grid(True, alpha=0.25)

plt.subplot(2, 2, 3)
plt.plot(x, hist_G)
plt.title("histograma G")
plt.xlim(0, 255)
plt.grid(True, alpha=0.25)

plt.subplot(2, 2, 4)
plt.plot(x, hist_B)
plt.title("histograma B")
plt.xlim(0, 255)
plt.grid(True, alpha=0.25)

plt.tight_layout()
plt.show()
