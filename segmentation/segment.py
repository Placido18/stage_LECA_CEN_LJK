import cv2
import numpy as np

image_path = "/Users/placideneuilly/Desktop/stage-neige/projgit/decoup/test/10-22-centre.jpg"

image = cv2.imread(image_path)
hauteur, _, _ = image.shape

# CLAHE pour égaliser

# LAB pour ne pas toucher aux couleurs
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)


clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Recomposition de l'image améliorée
limg = cv2.merge((cl, a, b))
image_egalisee = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# détection HSV
image_hsv = cv2.cvtColor(image_egalisee, cv2.COLOR_BGR2HSV)

snow_mask_sun = cv2.inRange(image_hsv, (0,0,180), (180, 50, 255))
snow_mask_dark = cv2.inRange(image_hsv, (75, 2, 85), (135, 160, 170))

final_mask = cv2.bitwise_or(snow_mask_sun, snow_mask_dark)

# on enlève le ciel
sky_limit = int(hauteur * 0.46)
final_mask[:sky_limit, :] = 0

cv2.imshow("image", image)
cv2.imshow("masque neige", final_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
