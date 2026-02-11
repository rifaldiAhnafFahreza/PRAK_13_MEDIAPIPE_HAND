import cv2  # import module opencv
import mediapipe

capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
hand = mediapipehand.Hands(max_num_hands=1)
mpdraw = mediapipe.solutions.drawing_utils

while True:
    success, frame = capture.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand.process(imgRGB)

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

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
