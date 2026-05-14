import mediapipe as mp
import cv2
import json
import math
import time

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


def _distance(point_a: dict, point_b: dict) -> float:
    return math.sqrt(
        (point_a['x'] - point_b['x']) ** 2 +
        (point_a['y'] - point_b['y']) ** 2 +
        (point_a['z'] - point_b['z']) ** 2
    )


def _average(values):
    return sum(values) / len(values) if values else 0.0


def _get_landmark(pose_data: dict, name: str) -> dict:
    return pose_data.get(name, {'x': 0.0, 'y': 0.0, 'z': 0.0, 'presence': 0.0})


def _vector(a: dict, b: dict) -> dict:
    return {'x': b['x'] - a['x'], 'y': b['y'] - a['y'], 'z': b['z'] - a['z']}


def _norm(vec: dict) -> float:
    return math.sqrt(vec['x'] ** 2 + vec['y'] ** 2 + vec['z'] ** 2)


def _angle_between(a: dict, b: dict) -> float:
    norm_a = _norm(a)
    norm_b = _norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = a['x'] * b['x'] + a['y'] * b['y'] + a['z'] * b['z']
    cos_theta = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.degrees(math.acos(cos_theta))


def _midpoint(a: dict, b: dict) -> dict:
    return {'x': (a['x'] + b['x']) / 2, 'y': (a['y'] + b['y']) / 2, 'z': (a['z'] + b['z']) / 2}


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
        capture_object = cv2.VideoCapture(1)
        if not capture_object.isOpened():
            raise RuntimeError('Camera not available')

        retval, frame = capture_object.read()
        capture_object.release()
        if not retval or frame is None:
            raise RuntimeError('Camera frame capture failed')

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print('captureframe')
        cv2.imwrite('pose_image.jpg', frame)
        return frame_rgb
    
    def _extract_pose_data(self, results) -> dict:
        if not results.pose_landmarks:
            return {}

        pose_data = {}
        for idx, landmark in enumerate(results.pose_landmarks[0]):
            name = POSE_LANDMARK_NAMES[idx]
            pose_data[name] = {
                'x': round(landmark.x, 3),
                'y': round(landmark.y, 3),
                'z': round(landmark.z, 3),
                'presence': round(landmark.presence, 3)
            }
        return pose_data

    def processframe(self) -> dict:
        with self.poselandmarker.create_from_options(self.options) as Landmarker:
            frame_rgp_target = self.captureframe()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgp_target)
            results = Landmarker.detect(mp_image)
        return self._extract_pose_data(results)

    def capture_sequence(self, count: int = 3, delay: float = 0.08) -> list[dict]:
        pose_sequence = []
        with self.poselandmarker.create_from_options(self.options) as Landmarker:
            for _ in range(count):
                frame_rgp_target = self.captureframe()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgp_target)
                results = Landmarker.detect(mp_image)
                pose_sequence.append(self._extract_pose_data(results))
                time.sleep(delay)
        return pose_sequence


class frame_classifier:
    def __init__(self):
        self.pose_processor = PoseProcessor('Helper_Models/pose_landmarker_lite.task')

    def get_face(self) -> dict:
        pose_data = self.pose_processor.processframe()
        return {name: pose_data[name] for name in POSE_LANDMARK_NAMES[:11] if name in pose_data}

    def get_body(self) -> dict:
        pose_data = self.pose_processor.processframe()
        return {name: pose_data[name] for name in POSE_LANDMARK_NAMES[11:] if name in pose_data}

    def get_whole_person(self) -> dict:
        return self.pose_processor.processframe()

    def get_sequence(self, count: int = 3) -> list[dict]:
        return self.pose_processor.capture_sequence(count=count)

    def compute_spatial_features(self, pose_sequence: list[dict]) -> dict:
        if not pose_sequence:
            return {}

        last_pose = pose_sequence[-1]
        required = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
        if any(name not in last_pose for name in required):
            return {}

        left_shoulder = _get_landmark(last_pose, 'LEFT_SHOULDER')
        right_shoulder = _get_landmark(last_pose, 'RIGHT_SHOULDER')
        left_hip = _get_landmark(last_pose, 'LEFT_HIP')
        right_hip = _get_landmark(last_pose, 'RIGHT_HIP')
        left_knee = _get_landmark(last_pose, 'LEFT_KNEE')
        right_knee = _get_landmark(last_pose, 'RIGHT_KNEE')

        avg_shoulder_y = _average([left_shoulder['y'], right_shoulder['y']])
        avg_hip_y = _average([left_hip['y'], right_hip['y']])
        avg_knee_y = _average([left_knee['y'], right_knee['y']])

        shoulder_dist = _distance(left_shoulder, right_shoulder)
        hip_dist = _distance(left_hip, right_hip)
        knee_dist = _distance(left_knee, right_knee)

        vertical_span = abs(avg_knee_y - avg_shoulder_y)
        spine_drop = abs(avg_hip_y - avg_shoulder_y)
        is_horizontal = abs(avg_shoulder_y - avg_hip_y) < 0.08 and abs(avg_hip_y - avg_knee_y) < 0.08

        all_depths = [value['z'] for value in last_pose.values() if 'z' in value]
        depth_spread = max(all_depths, default=0.0) - min(all_depths, default=0.0)

        movement_scores = []
        for i in range(1, len(pose_sequence)):
            prev_pose = pose_sequence[i - 1]
            curr_pose = pose_sequence[i]
            if not prev_pose or not curr_pose:
                continue
            frame_movements = []
            for name in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']:
                if name in prev_pose and name in curr_pose:
                    frame_movements.append(_distance(prev_pose[name], curr_pose[name]))
            if frame_movements:
                movement_scores.append(_average(frame_movements))

        movement_score = _average(movement_scores)
        start_shoulder_y = _average([
            _average([pose['LEFT_SHOULDER']['y'], pose['RIGHT_SHOULDER']['y']])
            for pose in pose_sequence[:1] if 'LEFT_SHOULDER' in pose and 'RIGHT_SHOULDER' in pose
        ])
        recent_drop = avg_shoulder_y - start_shoulder_y

        head = _get_landmark(last_pose, 'NOSE')
        mid_shoulder = _midpoint(left_shoulder, right_shoulder)
        mid_hip = _midpoint(left_hip, right_hip)
        torso_vector = _vector(mid_shoulder, mid_hip)
        torso_angle = _angle_between(torso_vector, {'x': 0.0, 'y': 1.0, 'z': 0.0})

        left_knee = _get_landmark(last_pose, 'LEFT_KNEE')
        right_knee = _get_landmark(last_pose, 'RIGHT_KNEE')
        left_ankle = _get_landmark(last_pose, 'LEFT_ANKLE')
        right_ankle = _get_landmark(last_pose, 'RIGHT_ANKLE')

        left_leg_angle = _angle_between(_vector(left_hip, left_knee), _vector(left_knee, left_ankle))
        right_leg_angle = _angle_between(_vector(right_hip, right_knee), _vector(right_knee, right_ankle))

        head_to_hip = _distance(head, mid_hip)
        head_hip_ratio = head_to_hip / vertical_span if vertical_span else 0.0
        shoulder_knee_ratio = shoulder_dist / knee_dist if knee_dist else 0.0
        hip_knee_ratio = hip_dist / knee_dist if knee_dist else 0.0
        head_slope = abs(head['x'] - mid_hip['x']) / vertical_span if vertical_span else 0.0

        return {
            'shoulder_distance': shoulder_dist,
            'hip_distance': hip_dist,
            'knee_distance': knee_dist,
            'vertical_span': vertical_span,
            'spine_drop': spine_drop,
            'is_horizontal': is_horizontal,
            'depth_spread': depth_spread,
            'movement_score': movement_score,
            'recent_drop': recent_drop,
            'shoulder_hip_height': abs(avg_shoulder_y - avg_hip_y),
            'torso_angle': torso_angle,
            'left_leg_angle': left_leg_angle,
            'right_leg_angle': right_leg_angle,
            'head_hip_ratio': head_hip_ratio,
            'shoulder_knee_ratio': shoulder_knee_ratio,
            'hip_knee_ratio': hip_knee_ratio,
            'head_slope': head_slope,
        }

# # testing the class
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
