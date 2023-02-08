import cv2
import numpy as np

a = int
l = int
x = int
y = int


classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostra = 25
identificador = input('Digite seu identificador: ')
altura = 220
largura = 220
print('Capturando as faces...')

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    print('Nivel da luz: ', np.average(imagemCinza))
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(40, 40))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
    if cv2.waitKey(27) & 0xFF == ord('q'):
        if np.average(imagemCinza) > 90:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite('Fotos/pessoa' + '.' + str(identificador) + '.' + str(amostra) + '.jpg', imagemFace)
            print('[foto' + str(amostra) + 'capturada com sucesso]')
            amostra += 1

    cv2.imshow('Face', imagem)
    cv2.waitKey(1)
    if amostra >= numeroAmostra + 1:
        break

print('Faces capturadas com sucesso!')
camera.release()

cv2.destroyAllWindows()
