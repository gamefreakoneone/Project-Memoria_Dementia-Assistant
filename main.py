import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if cap.isOpened():
    print("Second camera works!")
    cap.release()
else:
    print("Second camera failed")
    cap.release()