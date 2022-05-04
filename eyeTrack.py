import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [474, 475, 476, 477]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
             133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_IRIS = [469, 470, 471, 472]


mpFaceMesh = mp.solutions.face_mesh
with mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, img = cap.read()
        # Flip image frame
        # img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        results = face_mesh.process(imgRGB)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            mesh_points = np.array([np.multiply([p.x, p.y], [img_width, img_height]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            cv2.polylines(img, [mesh_points[LEFT_EYE]],
                          True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(img, [mesh_points[RIGHT_EYE]],
                          True, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.polylines(img, [mesh_points[LEFT_IRIS]],
                          True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.polylines(img, [mesh_points[RIGHT_IRIS]],
                          True, (0, 0, 255), 1, cv2.LINE_AA)

            (leftEye_cx, leftEye_cy), leftEye_radius = cv2.minEnclosingCircle(
                mesh_points[LEFT_IRIS])
            (rightEye_cx, rightEye_cy), rightEye_radius = cv2.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])

            centerLeft = np.array([leftEye_cx, leftEye_cy], dtype=np.int32)
            centerRight = np.array([rightEye_cx, rightEye_cy], dtype=np.int32)
            cv2.circle(img, centerLeft, int(leftEye_radius),
                       (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, centerRight, int(rightEye_radius),
                       (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Frame', img)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
