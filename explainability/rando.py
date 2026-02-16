import cv2

cap = cv2.VideoCapture("output/gradcam_video.mp4")
print("Frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
cap.release()