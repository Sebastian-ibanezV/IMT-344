import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("images.jpg")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

_, R_bin = cv2.threshold(R, 127, 255, cv2.THRESH_BINARY)
_, G_bin = cv2.threshold(G, 127, 255, cv2.THRESH_BINARY)
_, B_bin = cv2.threshold(B, 127, 255, cv2.THRESH_BINARY)

suma_bin = R_bin + G_bin + B_bin

suma_vis = cv2.normalize(suma_bin, None, 0, 255, cv2.NORM_MINMAX)

suma_logica = np.where((R_bin > 0) | (G_bin > 0) | (B_bin > 0), 255, 0).astype(np.uint8)

plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.imshow(R, cmap="gray")
plt.title("Canal R")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(G, cmap="gray")
plt.title("Canal G")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(B, cmap="gray")
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
plt.imshow(suma_vis, cmap="gray")
plt.title("Suma de binarios")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(suma_logica, cmap="gray")
plt.title("Union binaria")
plt.axis("off")

plt.tight_layout()
plt.show()