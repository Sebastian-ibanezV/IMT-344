import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("image.jpg")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

_, R_bin = cv2.threshold(R, 127, 255, cv2.THRESH_BINARY)
_, G_bin = cv2.threshold(G, 127, 255, cv2.THRESH_BINARY)
_, B_bin = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)

suma_bin = R_bin + G_bin + B_bin


plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(R)
plt.title("Canal R")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(G)
plt.title("Canal G")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(B)
plt.title("Canal B")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(R_bin, cmap="gray")
plt.title("R binario")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(G_bin, cmap="gray")
plt.title("G binario")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(B_bin, cmap="gray")
plt.title("B binario")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(suma_bin, cmap="gray")
plt.title("Suma de binarios")
plt.axis("off")

plt.tight_layout()
plt.show()
