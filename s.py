import cv2
import os

# Configuraciones
input_folder = 'tu_carpeta_original'  # Carpeta donde están las fotos de internet
output_folder = 'dataset_listo'       # Carpeta donde se guardarán procesadas
target_size = (64, 64)                # Tamaño común para PCA (ej. 64x64 o 128x128)

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f"Iniciando procesamiento en: {input_folder}...")

count = 0
for filename in os.listdir(input_folder):
    # Extensiones permitidas
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        try:
            img_path = os.path.join(input_folder, filename)
            
            # 1. Cargar la imagen
            img = cv2.imread(img_path)
            
            if img is None:
                continue

            # 2. Convertir a Grayscale (Escala de grises)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 3. Redimensionar (Interpolación AREA es mejor para reducir tamaño)
            resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

            # 4. Guardar en formato uniforme (PNG)
            new_filename = f"img_{count:03d}.png"
            cv2.imwrite(os.path.join(output_folder, new_filename), resized)
            
            count += 1
        except Exception as e:
            print(f"Error con {filename}: {e}")

print(f"¡Listo! Se procesaron {count} imágenes en '{output_folder}'.")
