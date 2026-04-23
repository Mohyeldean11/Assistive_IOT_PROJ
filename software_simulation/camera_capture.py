import cv2, os,time
import mediapipe as mp

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

# file_path = os.getcwd() + "pic.jpg"
#media pipe initialization
baseOptions = mp.tasks.BaseOptions
poseLandmarker = mp.tasks.vision.PoseLandmarker
poseLandmarkeroptions = mp.tasks.vision.PoseLandmarkerOptions
poseVisionRunningmode = mp.tasks.vision.RunningMode

options = poseLandmarkeroptions(base_options=baseOptions(model_asset_path = "Case_study\Assistive_IOT_PROJ\Helper_Models\pose_landmarker_lite.task"),
                                running_mode = poseVisionRunningmode.IMAGE)

def captureframes()->dict :

    captureobject = cv2.VideoCapture(0)
    if captureobject.isOpened() :
        retval,frame = captureobject.read()
    captureobject.release()

    # Convert and analyze
    with poseLandmarker.create_from_options(options) as landmarker:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect(mp_image)
        cv2.imwrite("Case_study\Assistive_IOT_PROJ\image.jpg",frame_rgb)

    # Check results
    if results.pose_landmarks:
        print("Person detected")
        pose_date = {}
        for idx,landmark in enumerate(results.pose_landmarks[0]):
            name = POSE_LANDMARK_NAMES[idx]
            pose_date[name] = {
                "x":round(landmark.x,3),
                "y":round(landmark.y,3),
                "z":round(landmark.z,3),
                "presence": round(landmark.presence , 3)
            }
    else:
        print("No person detected")
    print(pose_date)
    
def skeleton_process()->dict:
    pass


list_json = captureframes()