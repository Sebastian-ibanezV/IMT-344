# Mini demo: compresión JPEG (lossy) a distintas calidades
# - No guarda archivos: comprime en memoria (BytesIO)
# - Muestra: original + versiones JPEG + tamaños (KB) + PSNR (opcional)

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import math

IMG_PATH = "tu_imagen_color.jpg"   # <-- cambia esto
qualities = [95, 60, 30, 10]       # puedes cambiar (más bajo = más pérdida)

# ---------- helpers ----------
def jpeg_roundtrip(img_rgb_pil: Image.Image, quality: int):
    """Comprime a JPEG en memoria y vuelve a decodificar. Devuelve (img_decodificada, size_bytes)."""
    buf = BytesIO()
    img_rgb_pil.save(buf, format="JPEG", quality=quality, optimize=True)  # compresión lossy
    size_bytes = buf.tell()
    buf.seek(0)
    decoded = Image.open(buf).convert("RGB")  # vuelve a RGB para comparar/mostrar
    return decoded, size_bytes

def psnr(a: np.ndarray, b: np.ndarray):
    """PSNR simple (dB) para comparar calidad. a y b: uint8 RGB."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((255.0 ** 2) / mse)

# ---------- cargar imagen ----------
orig = Image.open(IMG_PATH).convert("RGB")
orig_np = np.array(orig)

# ---------- hacer roundtrips ----------
results = []
for q in qualities:
    dec, sz = jpeg_roundtrip(orig, q)
    dec_np = np.array(dec)
    results.append((q, dec, sz, psnr(orig_np, dec_np)))

# ---------- plot ----------
cols = len(qualities) + 1
plt.figure(figsize=(4 * cols, 5))

# original
plt.subplot(1, cols, 1)
plt.imshow(orig)
plt.title("Original")
plt.axis("off")

# versiones jpeg
for i, (q, dec, sz, p) in enumerate(results, start=2):
    kb = sz / 1024
    plt.subplot(1, cols, i)
    plt.imshow(dec)
    plt.title(f"JPEG q={q}\n{kb:.1f} KB | PSNR {p:.1f} dB")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ---------- extra opcional: ver “qué se perdió” (diferencia) ----------
# Descomenta si tu profe quiere que se vea el error visualmente
"""
plt.figure(figsize=(4 * cols, 5))
plt.subplot(1, cols, 1)
plt.imshow(np.zeros_like(orig_np))
plt.title("Diferencia (0)")
plt.axis("off")

for i, (q, dec, sz, p) in enumerate(results, start=2):
    diff = np.abs(orig_np.astype(np.int16) - np.array(dec).astype(np.int16)).astype(np.uint8)
    # amplificar para que se vea mejor (clipea en 255)
    diff_amp = np.clip(diff * 5, 0, 255).astype(np.uint8)
    plt.subplot(1, cols, i)
    plt.imshow(diff_amp)
    plt.title(f"Error x5\nq={q}")
    plt.axis("off")

plt.tight_layout()
plt.show()
"""

Si quieres que la demo sea aún más “wow” para clase, cambia qualities a algo como [100, 70, 40, 20, 10] y usa una foto con texturas (césped, hojas, cabello) porque ahí se nota más el efecto lossy.