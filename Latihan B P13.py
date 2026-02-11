import cv2 #improt module opencv
import mediapipe as mp

capture = cv2.VideoCapture(0)

mediapipehand = mp.solutions.hands
hand = mediapipehand.Hands(max_num_hands=1)
mpdraw = mp.solutions.drawing_utils

while True:
    success, frame = capture.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(imgRGB)

    if results.multi_hand_landmarks:
        for titiktangan in results.multi_hand_landmarks:

            mpdraw.draw_landmarks(frame,titiktangan,mediapipehand.HAND_CONNECTIONS)

            #Mendefinisikan Titik tanggan
            pergelangan = titiktangan.landmark[0]
            ujungjempol = titiktangan.landmark[4]
            ujungkelingking = titiktangan.landmark[20]

            h, w, c = frame.shape

            #koordinat pixel
            jempol = int(ujungjempol.x * w)
            kelingking = int(ujungkelingking.x * w)

            #cek psisi tanggan (depan atau belakang)
            if jempol > kelingking:
                posisi = "Tangan Belakang"
            else:
                posisi = "Telapak Tangan (depan)"

            cv2.putText(frame, posisi,(10,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()