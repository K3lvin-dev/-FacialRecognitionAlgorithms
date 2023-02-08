import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create(threshold=50)

# Caminho das Fotos

def getimgmComID():
    caminhos = [os.path.join('Fotos', f) for f in os.listdir('Fotos')]
    faces = []
    ids = []
    for caminhoimgm in caminhos:
        imgmFace = cv2.cvtColor(cv2.imread(caminhoimgm), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoimgm)[-1].split('.')[1])
        ids.append(id)
        faces.append(imgmFace)
    return np.array(ids), faces

ids, faces = getimgmComID()

print('Treinando...')

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento realizado')

