from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1) Cargar imagen en escala de grises (0..255)
img = np.array(Image.open("imagen.jpg").convert("L"), dtype=np.uint8)

L = 256
N = img.size

# 2) Histograma n_k
hist = np.bincount(img.ravel(), minlength=L).astype(np.float64)

# 3) PDF y CDF
pdf = hist / N
cdf = np.cumsum(pdf)

# 4) Ecualización (manual): s_k = (L-1)*CDF(k)
lut = np.round((L - 1) * cdf).astype(np.uint8)
eq = lut[img]

# 5) Mostrar resultados (2x2)
x = np.arange(L)

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.bar(x, hist, width=1.0)
plt.title("Histograma")
plt.xlim(0, 255)
plt.grid(True, alpha=0.2)


plt.subplot(2, 2, 4)
plt.imshow(eq, cmap="gray")
plt.title("Ecualizada")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(x, cdf)
plt.title("CDF")
plt.xlim(0, 255)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()