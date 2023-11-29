import cv2
import numpy as np

# Fungsi untuk membuat watermark
def create_watermark(x, y):
    # x = height, y = weight, k = scaling factor
    # Membuat watermark dengan ukuran x dan y
    # np.random.randint menghasilkan watermark berbasis pseudorandom berdasarkan waktu saat ini
    watermark = np.random.randint(2, size = (x, y)).astype(np.int16)
    # Mengubah nilai 0 menjadi -1
    watermark[watermark == 0] = -1
    return watermark

def embed_watermark(img, watermark):
    watermarkedimage = cv2.add(img, watermark)
    return watermarkedimage

def detect_watermark(watermark, watermarkedimage):
    watermark[watermark == -1] = 0
    watermark = watermark.astype(np.uint8)
    watermarkedimage = watermarkedimage.astype(np.uint8)
    correlation = cv2.matchTemplate(watermarkedimage, watermark, cv2.TM_CCORR)
    if correlation > 0.9:
        return True
    else:
        return False

# Menerima input gambar dan mengubahnya menjadi grayscale
img = cv2.imread('cat.jpeg', cv2.IMREAD_GRAYSCALE)
img = np.array(img, dtype=np.int16)

# Menerima input scaling factor
k = int(input("Masukkan scaling factor: "))

# Meneria input nama file hasil watermarking yang diinginkan
filename = input("Masukkan nama file hasil watermarking spasial: ")

x, y = img.shape[:2]
watermark = create_watermark(x, y) * k
watermarkedimage = embed_watermark(img, watermark)


# Menyimpan gambar hasil watermarking pada folder images
cv2.imwrite(f'images/{filename}.png', watermarkedimage)

# Mengecek apakah watermark berhasil ditanamkan
if detect_watermark(watermark, watermarkedimage):
    print("Watermark terdeteksi")
else:
    print("Watermark tidak terdeteksi")