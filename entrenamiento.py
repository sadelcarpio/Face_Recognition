import cv2
import os
import numpy as np

dataPath = '/home/sadelcarpio/AI_Projects/Face_Detector/Data' # ruta de los datos de entrenamiento
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes')

    for filename in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + filename)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' +filename, 0))
        image = cv2.imread(personPath + '/' + filename, 0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)

    label = label + 1

# print('etiquetas = ', labels)
# print('Número de etiquetas 0: ', np.count_nonzero(np.array(labels)==0))
# print('Número de etiquetas 1: ', np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Entrenando:
print('Entrenando ...')
face_recognizer.train(facesData, np.array(labels))

#Almacenar el modelo

face_recognizer.write('modeloEigenFace.xml')

print("Modelo guardado")

