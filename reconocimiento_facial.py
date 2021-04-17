import cv2
import os

dataPath = '/home/sadelcarpio/Code/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths= ', imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Leer el modelo
face_recognizer.read('modeloEigenFace.xml')

cap = cv2.VideoCapture('Prueba.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, flip_frame = cap.read()
    frame = cv2.flip(flip_frame, 1)
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces :
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        if result[1] < 7000:    
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA )
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA )
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)            


    cv2.imshow('frame', frame)

    k = cv2.waitKey(1) # 1 frame cada 1 ms
    if k == 27: #tecla esc rompe el bucle
        break

cap.release()
cv2.destroyAllWindows()