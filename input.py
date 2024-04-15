import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import os

import model
from model import ExerciseModel

BaseOptions: object = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = './pose_landmarker_full.task'

#print(os.path.isfile("C:/Users/HP/Downloads/Sem-2/COMP512 AOS/Project/my_model.keras"))

our_model = ExerciseModel("C:/Users/HP/Downloads/Sem-2/COMP512 AOS/Project/my_model.keras")
font = cv2.FONT_HERSHEY_SIMPLEX

mp_pose = mp.solutions.pose
# for lndmark in mp_pose.PoseLandmark:
#     print(lndmark)

RESULT = None


def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('pose landmarker result: {}'.format(result))
    global RESULT
    RESULT = result


def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    if angle > 180.0:
        angle = 360 - angle
    return angle


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # 0 indicates that the video should be taken from local webcam

    def __del__(self):
        self.video.release()

    def get_angels(self):
        with PoseLandmarker.create_from_options(options) as landmarker:
                _, fr = self.video.read()
                fr_np = np.array(fr)
                frame_timestamp_ms = int(self.video.get(cv2.CAP_PROP_POS_MSEC))
                # frame_timestamp_ms = int(round(time.time() * 1000))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=fr_np)
                landmarker.detect_async(mp_image, frame_timestamp_ms)
                try:
                    landmarks = RESULT.pose_landmarks[0]

                    # Extracting all left joints coordinates
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z]
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]

                    # Extracting all right joints coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                    mid_hip = [0, 0, 0]

                    right_elbow_right_shoulder_right_hip = calc_angle(right_elbow, right_shoulder, right_hip)
                    left_elbow_left_shoulder_left_hip = calc_angle(left_elbow, left_shoulder, left_hip)

                    right_knee_mid_hip_left_knee = calc_angle(right_knee, mid_hip, left_knee)

                    right_hip_right_knee_right_ankle = calc_angle(right_hip, right_knee, right_ankle)
                    left_hip_left_knee_left_ankle = calc_angle(left_hip, left_knee, left_ankle)

                    right_wrist_right_elbow_right_shoulder = calc_angle(right_wrist, right_elbow, right_shoulder)
                    left_wrist_left_elbow_left_shoulder = calc_angle(left_wrist, left_elbow, left_shoulder)

                    # To send to the server
                    input_data_stream_per_frame = [right_elbow_right_shoulder_right_hip,
                             left_elbow_left_shoulder_left_hip,
                             right_knee_mid_hip_left_knee,
                             right_hip_right_knee_right_ankle,
                             left_hip_left_knee_left_ankle,
                             right_wrist_right_elbow_right_shoulder,
                             left_wrist_left_elbow_left_shoulder]
                    # Need to send this list to server
                    # print(input_data_stream_per_frame)


                except:
                    pass
                # Annotating image
                if type(RESULT) is not type(None):
                    pose_landmarks_list = RESULT.pose_landmarks
                    for idx in range(len(pose_landmarks_list)):
                        pose_landmarks = pose_landmarks_list[idx]
                        # Draw the pose landmarks.
                        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        pose_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                            pose_landmarks
                        ])
                        solutions.drawing_utils.draw_landmarks(
                            fr,
                            pose_landmarks_proto,
                            solutions.pose.POSE_CONNECTIONS,
                            solutions.drawing_styles.get_default_pose_landmarks_style())

                        inp = pad_sequences(input_data_stream_per_frame, maxlen=350, padding='post', truncating='post')
                        pred = our_model.predict_exercise(inp)
                        cv2.putText(fr, pred, font, 1, (255, 255, 0),2)

                #cv2.imshow('Frame', fr)
                _, jpeg = cv2.imencode('.jpg', fr)
                return jpeg.tobytes()

