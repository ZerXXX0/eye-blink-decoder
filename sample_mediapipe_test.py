import sys
import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("camera_open_failed")
        sys.exit(2)

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("frame_read_failed")
        sys.exit(3)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as fm:
        results = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        print("no_faces")
    else:
        print("faces:", len(results.multi_face_landmarks))
        for i, lm in enumerate(results.multi_face_landmarks):
            print(f"face[{i}] landmarks:", len(lm.landmark))

    print("done")

if __name__ == '__main__':
    main()
