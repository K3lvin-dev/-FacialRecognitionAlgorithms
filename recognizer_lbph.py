import numpy as np
import cv2
import sys

detectorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read('classificadorLBPH.yml')
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_DUPLEX
camera = cv2.VideoCapture(0)


def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Detector inválido")
    sys.exit(1)


TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
BGS_TYPE = BGS_TYPES[2]


def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((3, 3), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

minArea = 250
bg_subtractor = getBGSubtractor(BGS_TYPE)

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id , confianca = reconhecedor.predict(imagemFace)
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
            cv2.putText(imagemFace, str('Desconhecido'), (x, y + (a+30)), font, 1, (0, 0, 255))

        imagem = imagemFace

        bg_mask = bg_subtractor.apply(imagem)
        bg_mask = getFilter(bg_mask, 'combine')
        bg_mask = cv2.medianBlur(bg_mask, 5)

        (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)


        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= minArea:
                x1, y1, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(imagem, (10, 30), (250, 55), (255, 0, 0), -1)
                cv2.putText(imagem, 'Movimento detectado!', (10, 50), FONT, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

                cv2.drawContours(imagem, cnt, -1, TRACKER_COLOR, 3)
                cv2.drawContours(imagem, cnt, -1, (255, 255, 255), 1)
                cv2.rectangle(imagem, (x1, y1), (x1 + w, y1 + h), TRACKER_COLOR, 3)
                cv2.rectangle(imagem, (x1, y1), (x1 + w, y1 + h), (255, 255, 255), 1)
        cv2.imshow('Frame', imagem)
        cv2.resizeWindow('Frame', 640, 480)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
