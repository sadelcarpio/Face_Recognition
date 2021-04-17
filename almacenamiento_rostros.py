import cv2
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

image = cv2.imread('imagen_000.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceClassif.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces: 

    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0,  255), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0) #al presionar Esc (cualquier tecla) se sale del loop

cv2.destroyAllWindows()