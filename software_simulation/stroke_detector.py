import math

class StrokeDetector:
    """
    Detects stroke warning signs using MediaPipe landmark asymmetry.
    Uses the FAST acronym: Face, Arms, Speech (we skip speech), Time.
    """

    ASYMMETRY_THRESHOLD = 0.03   # Lower threshold for better sensitivity
    ARM_DROP_THRESHOLD  = 0.05   # More sensitive arm detection
    COLLAPSE_THRESHOLD  = 0.10   # Sudden drop detection
    FREEZE_THRESHOLD    = 0.002  # Movement threshold for freeze detection

    def __init__(self):
        self.pose_history: list[dict] = []   # rolling window of last N frames
        self.WINDOW = 5

    def update(self, landmarks: dict):
        self.pose_history.append(landmarks)
        if len(self.pose_history) > self.WINDOW:
            self.pose_history.pop(0)

    def _get(self, landmarks, name, axis):
        return landmarks.get(name, {}).get(axis, None)

    # --- FACE: check mouth/eye asymmetry ---
    def check_facial_droop(self, landmarks: dict) -> bool:
        mouth_l = self._get(landmarks, 'MOUTH_LEFT',  'y')
        mouth_r = self._get(landmarks, 'MOUTH_RIGHT', 'y')
        eye_l   = self._get(landmarks, 'LEFT_EYE',    'y')
        eye_r   = self._get(landmarks, 'RIGHT_EYE',   'y')

        if None in (mouth_l, mouth_r, eye_l, eye_r):
            return False

        mouth_asymmetry = abs(mouth_l - mouth_r)
        eye_asymmetry   = abs(eye_l   - eye_r)
        return mouth_asymmetry > self.ASYMMETRY_THRESHOLD or \
               eye_asymmetry   > self.ASYMMETRY_THRESHOLD

    # --- ARMS: one arm hanging lower than the other ---
    def check_arm_weakness(self, landmarks: dict) -> bool:
        l_wrist = self._get(landmarks, 'LEFT_WRIST',  'y')
        r_wrist = self._get(landmarks, 'RIGHT_WRIST', 'y')
        if None in (l_wrist, r_wrist):
            return False
        return abs(l_wrist - r_wrist) > self.ARM_DROP_THRESHOLD

    # --- SUDDEN COLLAPSE: compare current frame to N frames ago ---
    def check_sudden_collapse(self) -> bool:
        if len(self.pose_history) < 2:
            return False
        oldest  = self.pose_history[0]
        current = self.pose_history[-1]

        old_nose = self._get(oldest,  'NOSE', 'y')
        new_nose = self._get(current, 'NOSE', 'y')

        if None in (old_nose, new_nose):
            return False

        # In MediaPipe, y increases downward — a collapse means nose y increases sharply
        drop = new_nose - old_nose
        return drop > self.COLLAPSE_THRESHOLD

    # --- POSE FREEZE: person stopped moving (possible loss of consciousness) ---
    def check_pose_freeze(self) -> bool:
        if len(self.pose_history) < self.WINDOW:
            return False
        # Compare shoulder positions across all frames
        movements = []
        for i in range(1, len(self.pose_history)):
            prev = self._get(self.pose_history[i-1], 'LEFT_SHOULDER', 'x')
            curr = self._get(self.pose_history[i],   'LEFT_SHOULDER', 'x')
            if None not in (prev, curr):
                movements.append(abs(curr - prev))
        avg_movement = sum(movements) / len(movements) if movements else 0
        return avg_movement < self.FREEZE_THRESHOLD  # nearly zero movement across window

    # --- GRADUAL DETERIORATION: check if pose is worsening over time ---
    def check_gradual_deterioration(self) -> bool:
        if len(self.pose_history) < self.WINDOW:
            return False
        # Check if shoulders are gradually dropping
        shoulder_ys = []
        for pose in self.pose_history:
            l_shoulder = self._get(pose, 'LEFT_SHOULDER', 'y')
            r_shoulder = self._get(pose, 'RIGHT_SHOULDER', 'y')
            if None not in (l_shoulder, r_shoulder):
                avg_shoulder_y = (l_shoulder + r_shoulder) / 2
                shoulder_ys.append(avg_shoulder_y)
        
        if len(shoulder_ys) < 3:
            return False
        
        # Check if there's a downward trend
        trend = shoulder_ys[-1] - shoulder_ys[0]
        return trend > 0.05  # Shoulders dropping over time

    def get_stroke_risk(self, landmarks: dict) -> dict:
        self.update(landmarks)
        facial_droop     = self.check_facial_droop(landmarks)
        arm_weakness     = self.check_arm_weakness(landmarks)
        sudden_collapse  = self.check_sudden_collapse()
        pose_freeze      = self.check_pose_freeze()
        gradual_deterioration = self.check_gradual_deterioration()

        signs_count = sum([facial_droop, arm_weakness, sudden_collapse, pose_freeze, gradual_deterioration])
        risk_level  = "HIGH" if signs_count >= 3 else ("MEDIUM" if signs_count >= 2 else "LOW")

        return {
            "risk_level":      risk_level,
            "facial_droop":    facial_droop,
            "arm_weakness":    arm_weakness,
            "sudden_collapse": sudden_collapse,
            "pose_freeze":     pose_freeze,
            "gradual_deterioration": gradual_deterioration,
            "signs_count":     signs_count
        }