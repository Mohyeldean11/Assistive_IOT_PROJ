import mediapipe as mp
import cv2 , json

POSE_LANDMARK_NAMES = [
    'NOSE',
    'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR',
    'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST',
    'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX',
    'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP',
    'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE',
    'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]


class PoseProcessor :
    def __init__(self,model_path:str ='../Helper_Models/pose_landmarker_lite.task' ):
        baseoptions = mp.tasks.BaseOptions
        self.poselandmarker = mp.tasks.vision.PoseLandmarker
        poselandmarker_options = mp.tasks.vision.PoseLandmarkerOptions
        poseVision_Running = mp.tasks.vision.RunningMode
        self.options = poselandmarker_options(base_options=baseoptions(model_asset_path = model_path),
                                running_mode = poseVision_Running.IMAGE)
        print('init')
        
        


    def captureframe(self):
        capture_object = cv2.VideoCapture(0)
        if capture_object.isOpened():
            retval,frame = capture_object.read()
            capture_object.release()
        frame_rgp = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        print('captureframe')
        cv2.imwrite('pose_image.jpg',frame)
        return frame_rgp
    
    def processframe(self)->dict:
        with self.poselandmarker.create_from_options(self.options) as Landmarker:
            frame_rgp_target = self.captureframe()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgp_target)
            results = Landmarker.detect(mp_image)
        print('processframe')
        if results.pose_landmarks:
            print('Person detected')
            pose_data = {}
            for idx,landmark in enumerate(results.pose_landmarks[0]):
                name = POSE_LANDMARK_NAMES[idx]
                pose_data[name] = {
                    'x':round(landmark.x,3),
                    'y':round(landmark.y,3),
                    'z':round(landmark.z,3),
                    'presence': round(landmark.presence , 3)
                }
        else:
            print('No person detected')
            pose_data = {}
        return pose_data


class frame_classifier : 
    def __init__(self):
        pose_obj = PoseProcessor('../Helper_Models/pose_landmarker_lite.task')
        self.pose_body_parts = pose_obj.processframe()
        

    def get_face(self)-> dict :
        face_pose = {}
        idx = 0
        for value in POSE_LANDMARK_NAMES[:11]:
            face_pose[POSE_LANDMARK_NAMES[idx]] = self.pose_body_parts[value]
            idx += 1
        return face_pose
    
    def get_body(self)-> dict :
        body_pose = {}
        idx = 11
        for values in POSE_LANDMARK_NAMES[11:]:
            body_pose[POSE_LANDMARK_NAMES[idx]] = self.pose_body_parts[values]
            idx += 1
        return body_pose

    def get_whole_person(self) -> dict :
        whole_person = self.get_face() | self.get_body()
        return whole_person

"""testing the class
# this_dict = PoseProcessor_instance.processframe()
# for values in POSE_LANDMARK_NAMES:
#     print(this_dict[values])
#     print('\n')

# frame_classifier_inst = frame_classifier()
# tempdict1 = frame_classifier_inst.get_face()
# with open("test1.json","a") as fs:
#     for key , value in tempdict1.items():
#         fs.write(f'face part : {key} : {value} \n')

# tempdict2 = frame_classifier_inst.get_body()
# for key , value in tempdict2.items():
#     print(f' body part : {key } : {value}\n')
# print(POSE_LANDMARK_NAMES)

# frame_classifier_inst = frame_classifier()
# tempdict3 =frame_classifier_inst.get_whole_person()
# with open("finaltest.txt","a") as fs:
#     for key , item in tempdict3.items():
#         fs.write(f'the part {key} : {item} \n')
"""