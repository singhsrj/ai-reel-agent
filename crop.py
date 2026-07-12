"""
Sample a few frames from the source video, detect the speaker's face, and
compute a single static x-offset for a 9:16 crop that keeps them centered.
MVP scope: static crop, no per-frame panning.
"""
import statistics
import cv2


def detect_face_center_x(video_path: str, sample_times: list, frame_width: int) -> float:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(video_path)

    centers = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])  # largest face
            centers.append(fx + fw / 2)
    cap.release()

    if not centers:
        return frame_width / 2  # fallback: dead center
    return statistics.median(centers)


def compute_crop_x(center_x: float, crop_width: int, frame_width: int) -> int:
    x = int(center_x - crop_width / 2)
    return max(0, min(x, frame_width - crop_width))
