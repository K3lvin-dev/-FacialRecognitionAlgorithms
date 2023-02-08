import cv2
import numpy as np

detectorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read('classificadorFisher.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_DUPLEX
camera = cv2.VideoCapture('a007.mkv')

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ''
        if id == 1:
            nome = 'Kid01'
        elif id == 2:
            nome = 'Kid02'
        elif id == 3:
            nome = 'Kid03'
        elif id == 4:
            nome = 'Kid04'
        elif id == 5:
            nome = 'Kid05'
        elif id == 6:
            nome = 'Kid06'
        else:
            nome = 'Desconhecido'

        if nome == 'Desconhecido':
            cv2.putText(imagem, str(nome), (x, y + (a + 30)), font, 1, (0, 0, 255))
        else:
            confianca = round(confianca, 2)
            cv2.putText(imagem, str(nome), (x, y + (a+30)), font, 1, (0, 0, 255))
            cv2.putText(imagem, str(confianca), (x, y + (a+60)), font, 1, (0, 255, 0, 0))
    cv2.namedWindow('face', cv2.WINDOW_NORMAL)
    cv2.moveWindow('face', 0, 0)
    cv2.imshow('face', imagem)
    cv2.resizeWindow('face', 1920, 1080)
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()
