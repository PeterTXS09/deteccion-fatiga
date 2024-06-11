import cv2
import mediapipe as mp
import math
import time

video = cv2.VideoCapture() # leer la cámara 0
mpFaceMesh = mp.solutions.face_mesh # instancia de lectura de rostro
faceMesh = mpFaceMesh.FaceMesh() # implementando la lectura de rostros
mpDraw = mp.solutions.drawing_utils # dibujar los puntos
estado = 'X' # <--- Estado por defecto
inicio = 0 # tiempo en segundos
estado_actual = '' # en qué estado estoy

while True:
    check, img = video.read() # leer la webcam
    img = cv2.resize(img, (1000, 720)) # 1000, 720
    if not check:
        break
    results = faceMesh.process(img) # img (el fotograma actual) está siendo procesada por el lector de rostros
    h, w, _ = img.shape

    if results:
        if not results.multi_face_landmarks: # si no detecta nada, se salta el frame
            continue
        for face in results.multi_face_landmarks: # si detecta...
            #print(face)
            # mpDraw.draw_landmarks(img, face, mpFaceMesh.FACEMESH_FACE_OVAL)
            d1x, d1y = int((face.landmark[159].x)*w), int((face.landmark[159].y)*h) # izquierdo arriba
            d2x, d2y = int((face.landmark[145].x) * w), int((face.landmark[145].y) * h) # izq. abajo
            i1x, i1y = int((face.landmark[386].x) * w), int((face.landmark[386].y) * h) # der. arriba
            i2x, i2y = int((face.landmark[374].x) * w), int((face.landmark[374].y) * h) # der. abajo

            distD = math.hypot(d1x - d2x, d1y - d2y) # hallar la distancia entre los puntos
            distI = math.hypot(i1x - i2x, i1y - i2y)


            print(f'distD: {distD}, distI: {distI}')
            if distI <= 19 and distD <= 19:
                print('ojos cerrados')
                cv2.rectangle(img, (100, 30), (390, 80), (0,0,255), -1)
                cv2.putText(img, 'OJOS CERRADOS', (105,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255, 255), 3)
                estado = 'Dormido'
                if estado != estado_actual:
                    inicio = time.time() # desde qué segundo está dormido
            else:
                print('ojos abiertos')
                cv2.rectangle(img, (100, 30), (390, 80), (255, 0, 0), -1)
                cv2.putText(img, 'OJOS ABIERTOS', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                estado = 'Despierto'
                inicio = time.time()
                tiempo = int(time.time() - inicio)

            if estado == 'Dormido':
                tiempo = int(time.time() - inicio) # tomar el tipo

            if tiempo >= 2: # si es mayor a 2 segundos
                cv2.rectangle(img, (300, 150), (850, 220), (0,0,255), -1)
                cv2.putText(img, f'DORMIDO: {tiempo} SEG', (310, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 5)
            estado_actual = estado
    cv2.imshow('Detector', img)
    cv2.waitKey(10)
