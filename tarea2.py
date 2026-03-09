import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("image.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

R_img = np.zeros_like(img_rgb)
G_img = np.zeros_like(img_rgb)
B_img = np.zeros_like(img_rgb)

R_img[:, :, 0] = R
G_img[:, :, 1] = G
B_img[:, :, 2] = B

_, R_bin = cv2.threshold(R, 127, 255, cv2.THRESH_BINARY)
_, G_bin = cv2.threshold(G, 127, 255, cv2.THRESH_BINARY)
_, B_bin = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)

suma_bin = R_bin + G_bin + B_bin

plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(R_img)
plt.title("Capa R")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(G_img)
plt.title("Capa G")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(B_img)
plt.title("Capa B")
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

plt.figure(figsize=(6, 4))
plt.imshow(suma_bin, cmap="gray")
plt.title("Suma de binarios")
plt.axis("off")
plt.tight_layout()
plt.show()