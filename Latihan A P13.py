import cv2
import mediapipe

capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
hand = mediapipehand.Hands(max_num_hands=1)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, frame = capture.read(-1)
    if not success:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(imgRGB)

    # DETEKSI LANDMARK
    if results.multi_hand_landmarks:
        for titiktangan in results.multi_hand_landmarks:

            mpdraw.draw_landmarks(
                frame,
                titiktangan,
                mediapipehand.HAND_CONNECTIONS
            )

            for id, titik in enumerate(titiktangan.landmark):
                print(id)
                print(titik.x)
                print(titik.y)

    # KLASIFIKASI TANGAN KANAN / KIRI
    if results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):

            label = hand_handedness.classification[0].label

            if label == "Right":
                cv2.putText(frame, "Tangan Kiri",
                            (20, 50),
                            cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0), 3)

            elif label == "Left":
                cv2.putText(frame, "Tangan Kanan",
                            (20, 50),
                            cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
