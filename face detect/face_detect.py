import cv2 as cv
# read image
img =cv.imread('1.jpg')

# convert image to gray scale
gray_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# define haar cascade 
haar_cascade= cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# get a rectangle outlining a detected face -> ( X position, Y position, Width, Height)
facebox = haar_cascade.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5)

for x,y,w,h in  (facebox):
    cv.rectangle(img, (x,y),(x+w, y+h), color=(0,255,0), thickness= 2)



cv.imshow('Face', img)
cv.waitKey(0)