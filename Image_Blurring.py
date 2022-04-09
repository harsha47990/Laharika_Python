import cv2 as cv

img = cv.imread(r'D:\others\images\face.jpg')
blur = cv.bilateralFilter(img,-1,15,15)
cv.imwrite(r'D:\others\images\face1.jpg',blur)

dst = cv.fastNlMeansDenoisingColored(img,None,3,10,7,21)
cv.imwrite(r'D:\others\images\face1.jpg',dst)

median = cv.medianBlur(img,3)
cv.imwrite(r'D:\others\images\noise1.jpg',median)
