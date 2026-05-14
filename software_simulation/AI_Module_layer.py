import ollama, sensorfusion, Mediapipe_Class
import stroke_detector
import pose_training

poses = ("INIT","SITTING","STANDING","LAYING ON THE FLOOR","LAYING ON THE BED","FALLING","COLLAPSING")
Person_status = ('Normal','Not Normal','Not Found')

class AI_LAYER:
    def __init__(self, model: str = "llama3.2", model_path: str = 'pose_model.pkl'):
        self.model = model
        self.old_pose = poses[0]
        self.person_pose = poses[0]
        self.frame_classifier_instance = Mediapipe_Class.frame_classifier()
        self.stroke_detector = stroke_detector.StrokeDetector()
        self.pose_model = pose_training.load_model(model_path)

        

    def Prompt_builder(self)->str :
        body_parts = self.frame_classifier_instance.get_whole_person()
        prompt = f"""You are a medical AI assistant specializing in pose analysis for stroke detection. Analyze the given body landmark coordinates from MediaPipe pose detection.

                    Key stroke indicators to check:
                    - Facial droop: asymmetry in mouth or eyes
                    - Arm weakness: one arm significantly lower or asymmetric
                    - Sudden collapse: rapid drop in body position
                    - Loss of balance: leaning or falling poses

                    Based on the landmark data, classify the current pose as exactly ONE word from: SITTING, STANDING, LAYING_ON_FLOOR, LAYING_ON_BED, FALLING, or COLLAPSING.

                    Previous pose was: {self.old_pose}

                    Respond with ONLY the pose name, no explanation."""

        full_prompt = prompt + f"\nBody landmarks: {body_parts}\nRespond with one word:"
        return full_prompt
    
    def _classify_pose_from_rules(self, features: dict) -> str:
        if not features:
            return poses[0]

        horizontal = features.get('is_horizontal', False)
        movement = features.get('movement_score', 0.0)
        drop = features.get('recent_drop', 0.0)
        shoulder_hip = features.get('shoulder_hip_height', 0.0)
        depth_spread = features.get('depth_spread', 0.0)
        vertical_span = features.get('vertical_span', 0.0)

        if horizontal and movement > 0.03 and drop > 0.03:
            return 'FALLING'
        if horizontal and movement < 0.02:
            if depth_spread < 0.08:
                return 'LAYING ON THE BED'
            return 'LAYING ON THE FLOOR'
        if shoulder_hip > 0.18 and vertical_span > 0.24:
            return 'STANDING'
        if 0.08 < shoulder_hip <= 0.18:
            return 'SITTING'

        return 'STANDING'

    def _predict_pose_from_model(self, features: dict) -> str | None:
        if self.pose_model is None or not features:
            return None
        try:
            prediction = pose_training.predict_from_features(features, self.pose_model)
            return str(prediction)
        except Exception:
            return None

    def Get_Elder_Pose(self) -> str :
        self.old_pose = getattr(self, "person_pose", poses[0])
        pose_sequence = self.frame_classifier_instance.get_sequence(count=4)
        features = self.frame_classifier_instance.compute_spatial_features(pose_sequence)
        current_pose = self._predict_pose_from_model(features)
        if current_pose is None:
            current_pose = self._classify_pose_from_rules(features)
        self.person_pose = current_pose
        self.old_pose = self.person_pose
        return self.person_pose
    
    def Get_Stroke_Risk(self) -> dict:
        landmarks = self.frame_classifier_instance.get_whole_person()
        return self.stroke_detector.get_stroke_risk(landmarks)
    
    def Get_Elder_status(self,sensor_grp_data : dict )->str :
        final_prompt = f"""Analyze sensor data for elderly health monitoring. Sensor data: {str(sensor_grp_data)}

                            Classify as exactly ONE word: 'Normal', 'Not_Normal', or 'Not_Found'.

                            Consider:
                            - Abnormal readings indicate 'Not_Normal'
                            - No data or invalid readings: 'Not_Found'

                            Respond with ONLY one word."""
        AI_person_status = ollama.generate (self.model , prompt= final_prompt)
        sensorfusion.CurrentPAYLOAD["Person_status"] = AI_person_status.response.strip()
        return AI_person_status.response.strip()
    
# # testing    
# AI_INST = AI_LAYER()
# with open("texttryai.txt", "a") as fs:
#     fs.write(AI_INST.Get_Elder_Pose())
