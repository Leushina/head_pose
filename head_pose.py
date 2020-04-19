#!/usr/bin/env python

import numpy as np
import argparse
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def estimate_pose(im, image_points, model_points, camera_matrix):
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector,
                                                     camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(im, p1, p2, (255, 0, 0), 2)

    # Display image
    # cv2.imshow("Output", im)
    # cv2.waitKey(0)
    return im


def find_face(frame):
    """
    Estimating face position in the frame with dlib
    :param frame: image frame from video
    :return: face and transformed image
    """
    img = np.array(frame)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(img, 0)
    return face, img


def find_landmarks(img, face):
    """
    FACIAL_LANDMARKS_IDXS = {
    "left_corner_mouth": 49,
    "right_corner_mouth": 55,
    "left_eye": 37,
    "right_eye": 46,
    "nose": 31,
    "chin": 9
    }
    Find coordinates of above landmarks with dlib pretrained model
    :param img: frame image transformed for finding landmarks
    :param face: found face from the frame
    :return: np.array - landmarks for listed above points (x, y)
    """

    landmarks = predictor(img, face)
    landmarks_list = []
    # all indexes are (value - 1)
    idx = [30, 8, 36, 45, 48, 54]
    for i in idx:
        # save landmark coordinates
        x, y = landmarks.part(i).x, landmarks.part(i).y
        x, y = float(x), float(y)
        landmarks_list.append((x, y))

    return np.array(landmarks_list)


def process_video(camera_port=0):
    """
    Opens a window with live video. Estimate head pose
    :param camera_port: == (0 or -1 if webcam from laptop) or (videofile name)
    :return:
    """
    video_capture = cv2.VideoCapture(camera_port)
    model_points = np.array([

        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])
    camera_matrix = None
    # print("Camera Matrix :\n {0}".format(camera_matrix))
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret:
            if type(camera_matrix) != np.ndarray:
                size = frame.shape
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )
            if camera_port == 0:
                frame = cv2.flip(frame, 2)

            face, img = find_face(frame)
            #  img - transformed frame for later work
            if len(face) > 0:
                # if found face in the frame
                landmarks_list = find_landmarks(img, face[0])
                frame = estimate_pose(frame, landmarks_list, model_points, camera_matrix)

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()

if __name__ == "__main__":
    video_src = args.cam if args.cam is not None else args.video
    process_video(video_src)



