import mediapipe as mp
import cv2

POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]


class PoseProcessor :
    def __init__(self,model_path:str ="Helper_Models/pose_landmarker_lite.task" ):
        baseoptions = mp.tasks.BaseOptions
        self.poselandmarker = mp.tasks.vision.PoseLandmarker
        poselandmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        poseVision_Running = mp.tasks.vision.RunningMode
        self.options = poselandmarker_options(base_options=baseoptions(model_asset_path = model_path),
                                running_mode = poseVision_Running.IMAGE)
        


    def captureframe(self):
        capture_object = cv2.VideoCapture(0)
        if capture_object.isOpened():
            retval,frame = capture_object.read()
            capture_object.release()
        frame_rgp = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        cv2.imwrite("pose_image.jpg",frame)
        return frame_rgp
    
    def processframe(self,frame_rgp_target)->dict:
        with self.poselandmarker.create_from_options(self.options) as Landmarker:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgp_target)
            results = Landmarker.detect(mp_image)
        if results.pose_landmarks:
            print("Person detected")
            pose_data = {}
            for idx,landmark in enumerate(results.pose_landmarks[0]):
                name = POSE_LANDMARK_NAMES[idx]
                pose_data[name] = {
                    "x":round(landmark.x,3),
                    "y":round(landmark.y,3),
                    "z":round(landmark.z,3),
                    "presence": round(landmark.presence , 3)
                }
        else:
            print("No person detected")
            pose_data = {}
        return pose_data

PoseProcessor_instance = PoseProcessor()