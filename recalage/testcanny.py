import cv2

img_snow = cv2.imread("/Users/placideneuilly/Desktop/stage-neige/data/echant-juin-octobre-2023/2023-10-22-10-00-06-001.jpg")
img_dark = cv2.imread("/Users/placideneuilly/Desktop/stage-neige/data/echant-juin-octobre-2023/2023-09-19-18-00-05-001.jpg")

snow_gray = cv2.cvtColor(img_snow, cv2.COLOR_BGR2GRAY)
dark_gray = cv2.cvtColor(img_dark, cv2.COLOR_BGR2GRAY)


contours_snow = cv2.Canny(snow_gray, 30, 50, apertureSize=3, L2gradient=True)
contours_dark = cv2.Canny(dark_gray, 100, 250, apertureSize=5)

cv2.imshow("neige", contours_dark)
cv2.waitKey(0)
cv2.destroyAllWindows()
