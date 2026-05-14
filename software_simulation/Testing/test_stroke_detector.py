import unittest
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from software_simulation.stroke_detector import StrokeDetector

class TestStrokeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = StrokeDetector()

    def test_initialization(self):
        self.assertEqual(len(self.detector.pose_history), 0)
        self.assertEqual(self.detector.WINDOW, 5)

    def test_update_pose_history(self):
        landmarks = {'NOSE': {'x': 0.5, 'y': 0.2, 'z': 0.0}}
        self.detector.update(landmarks)
        self.assertEqual(len(self.detector.pose_history), 1)
        self.assertEqual(self.detector.pose_history[0], landmarks)

    def test_update_pose_history_window_limit(self):
        # Add more than WINDOW poses
        for i in range(7):
            landmarks = {'NOSE': {'x': 0.5, 'y': 0.2 + i*0.01, 'z': 0.0}}
            self.detector.update(landmarks)
        self.assertEqual(len(self.detector.pose_history), 5)  # Should maintain window size

    def test_check_facial_droop_no_asymmetry(self):
        landmarks = {
            'MOUTH_LEFT': {'y': 0.7},
            'MOUTH_RIGHT': {'y': 0.7},
            'LEFT_EYE': {'y': 0.6},
            'RIGHT_EYE': {'y': 0.6}
        }
        result = self.detector.check_facial_droop(landmarks)
        self.assertFalse(result)

    def test_check_facial_droop_with_asymmetry(self):
        landmarks = {
            'MOUTH_LEFT': {'y': 0.7},
            'MOUTH_RIGHT': {'y': 0.75},  # Asymmetry
            'LEFT_EYE': {'y': 0.6},
            'RIGHT_EYE': {'y': 0.6}
        }
        result = self.detector.check_facial_droop(landmarks)
        self.assertTrue(result)

    def test_check_arm_weakness_no_weakness(self):
        landmarks = {
            'LEFT_WRIST': {'y': 0.8},
            'RIGHT_WRIST': {'y': 0.8}
        }
        result = self.detector.check_arm_weakness(landmarks)
        self.assertFalse(result)

    def test_check_arm_weakness_with_weakness(self):
        landmarks = {
            'LEFT_WRIST': {'y': 0.8},
            'RIGHT_WRIST': {'y': 0.75}  # Right arm lower
        }
        result = self.detector.check_arm_weakness(landmarks)
        self.assertTrue(result)

    def test_check_sudden_collapse_no_history(self):
        result = self.detector.check_sudden_collapse()
        self.assertFalse(result)

    def test_check_sudden_collapse_with_drop(self):
        # Add initial pose
        self.detector.update({'NOSE': {'y': 0.5}})
        # Add pose with sudden drop
        self.detector.update({'NOSE': {'y': 0.65}})  # Nose moved down
        result = self.detector.check_sudden_collapse()
        self.assertTrue(result)

    def test_check_pose_freeze_no_movement(self):
        # Add multiple identical poses (no movement)
        for _ in range(5):
            self.detector.update({'LEFT_SHOULDER': {'x': 0.5}})
        result = self.detector.check_pose_freeze()
        self.assertTrue(result)

    def test_check_pose_freeze_with_movement(self):
        # Add poses with movement
        for i in range(5):
            self.detector.update({'LEFT_SHOULDER': {'x': 0.5 + i*0.01}})
        result = self.detector.check_pose_freeze()
        self.assertFalse(result)

    def test_get_stroke_risk_low(self):
        landmarks = {
            'MOUTH_LEFT': {'y': 0.7},
            'MOUTH_RIGHT': {'y': 0.7},
            'LEFT_EYE': {'y': 0.6},
            'RIGHT_EYE': {'y': 0.6},
            'LEFT_WRIST': {'y': 0.8},
            'RIGHT_WRIST': {'y': 0.8}
        }
        result = self.detector.get_stroke_risk(landmarks)
        self.assertEqual(result['risk_level'], 'LOW')
        self.assertEqual(result['signs_count'], 0)

    def test_get_stroke_risk_high(self):
        # Setup poses with multiple signs
        self.detector.update({'NOSE': {'y': 0.5}})
        self.detector.update({'NOSE': {'y': 0.65}})  # Sudden collapse
        self.detector.update({'NOSE': {'y': 0.66}})
        self.detector.update({'NOSE': {'y': 0.67}})
        self.detector.update({'NOSE': {'y': 0.68}})

        landmarks = {
            'MOUTH_LEFT': {'y': 0.7},
            'MOUTH_RIGHT': {'y': 0.75},  # Facial droop
            'LEFT_EYE': {'y': 0.6},
            'RIGHT_EYE': {'y': 0.6},
            'LEFT_WRIST': {'y': 0.8},
            'RIGHT_WRIST': {'y': 0.75},  # Arm weakness
            'LEFT_SHOULDER': {'x': 0.5}  # For freeze check
        }
        result = self.detector.get_stroke_risk(landmarks)
        self.assertEqual(result['risk_level'], 'HIGH')
        self.assertGreaterEqual(result['signs_count'], 3)

if __name__ == '__main__':
    unittest.main()