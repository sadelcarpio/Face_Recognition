import cv2
import os
import imutils

personName = 'Sergio' #Objetivo del cual se extraerán imágenes
dataPath = '/home/sadelcarpio/AI_Projects/Face_Detector/Data' #Ruta de los datos de entrenamiento
personPath = dataPath + '/' + personName

if not os.path.exists(personPath): #Si no existe la ruta, crearla
    print('Carpeta creada: ', personPath)
    os.makedirs(personPath)

cap = cv2.VideoCapture('Sergio.mp4') #video del cual sacar los datos

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
count = 0

while True:
    ret, frame = cap.read()
    if ret == False: break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,  0), 2)
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150,150), interpolation=cv2.INTER_CUBIC)
        rostro_flip = cv2.flip(rostro,1)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro_flip) #aumentando datos
        count = count + 1
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 600: #captura 600 frames
        break

cap.release()
cv2.destroyAllWindows()