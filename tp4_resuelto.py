import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ----------------------------------------
# UTILIDADES
# ----------------------------------------

def load(path):
    return cv2.imread(path)

def show_match(img1, img2, matches, kp1, kp2, title):
    result = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(14, 6))
    plt.imshow(result[..., ::-1])
    plt.title(title)
    plt.axis("off")
    plt.show()


def sift_match(img1, img2, show=True, title=""):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=40)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if show:
        show_match(img1, img2, good, kp1, kp2, f"{title} - Coincidencias: {len(good)}")

    return len(good)


def add_gaussian_noise(img, sigma):

    ruido = np.random.normal(0, sigma, img.shape).astype(np.float32)
    ruidosa = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    return ruidosa


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def translate(img, tx, ty):
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# ----------------------------------------
# RUTAS
# ----------------------------------------

BASE = "ImagenesApropiadasTP4"

# imágenes del arco del triunfo
IMG_ARCO_REF = os.path.join(BASE, "__192x192__arc_0nuevo.png")
IMG_ARCO_TEST = os.path.join(BASE ,"__192x192__arc_1nuevo.png")

## imágenes principales
IMG_REF = os.path.join(BASE, "Alonso_El pintor caminante.jpg")
IMG_TEST1 = os.path.join(BASE, "ElCaminanteAlonso.jpg")
IMG_TEST2 = os.path.join(BASE, "ElCaminanteAlonso2.jpg")
IMG_TEST3 = os.path.join(BASE, "ElCaminanteAlonso3.jpg")

# ejemplos manipulados
IMG_REF_AUTO = os.path.join(BASE, "img_7025.png")
IMG_ROT = os.path.join(BASE, "2drotationExample.png")
IMG_BLUR = os.path.join(BASE, "blurExample.png")
IMG_ILL = os.path.join(BASE, "illuminationExample.png")
# ----------------------------------------
# CARGA
# ----------------------------------------

ref_arc = load(IMG_ARCO_REF)
test_arc = load(IMG_ARCO_TEST)

ref = load(IMG_REF)
t1 = load(IMG_TEST1)
t2 = load(IMG_TEST2)
t3 = load(IMG_TEST3)

ref_auto = load(IMG_REF_AUTO)
rot_ex = load(IMG_ROT)
blur_ex = load(IMG_BLUR)
ill_ex = load(IMG_ILL)

# ----------------------------------------
# EJECUCIÓN TP4
# ----------------------------------------

print("\n============================")
print(" TP4 — RESULTADOS SIFT")
print("============================\n")

resultados = {}

# A. Misma imagen con distintas transformaciones
resultados["Arco del triunfo"] = sift_match(ref_arc, test_arc, title="Arco del triunfo")

resultados["Alonso vs Test1"] = sift_match(ref, t1, title="Alonso vs Test1")
resultados["Alonso vs Test2"] = sift_match(ref, t2, title="Alonso vs Test2")
resultados["Alonso vs Test3"] = sift_match(ref, t3, title="Alonso vs Test3")

# Ejemplos del repo
resultados["Rotación fuerte"] = sift_match(ref, rot_ex, title="Rotación")
resultados["Blur (Desenfoque)"] = sift_match(ref, blur_ex, title="Blur")
resultados["Iluminación"] = sift_match(ref, ill_ex, title="Iluminación")

resultados["Imagen auto vs Rotación"] = sift_match(ref_auto, rot_ex, title="Imagen auto vs Rotación")
resultados["Imagen auto vs Blur"] = sift_match(ref_auto, blur_ex, title="Imagen auto vs Blur")
resultados["Imagen auto vs Iluminación"] = sift_match(ref_auto, ill_ex, title="Imagen auto vs Iluminación") 


# B. Agregar ruido a una imagen
#ruido1 = add_gaussian_noise(ref, 25)
ruido1 = add_gaussian_noise(ref, sigma=25.0)
resultados["Ruido en una imagen"] = sift_match(ref, ruido1, title="Ruido en 1 imagen")

# C. Agregar ruido en ambas
ruido2 = add_gaussian_noise(ref, sigma=25)
resultados["Ruido en ambas"] = sift_match(ruido1, ruido2, title="Ruido en ambas imágenes")

# D. Transformaciones extra
rot45 = rotate(ref, 45)
resultados["Rotación 45°"] = sift_match(ref, rot45, title="Rotación 45°")

trans = translate(ref, 50, 80)
resultados["Traslación"] = sift_match(ref, trans, title="Traslación")

scale = cv2.resize(ref, None, fx=0.5, fy=0.5)
resultados["Escala 0.5"] = sift_match(ref, scale, title="Escala 0.5")


# ----------------------------------------
# RESUMEN FINAL
# ----------------------------------------

print("\n\n===== RESUMEN FINAL DEL TP4 =====\n")
for k, v in resultados.items():
    print(f"{k:<3} → {v} coincidencias")

print("\nTP4 COMPLETADO.\n")
