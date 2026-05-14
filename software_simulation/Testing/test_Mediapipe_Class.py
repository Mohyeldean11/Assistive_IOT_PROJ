import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Mediapipe_Class import PoseProcessor, frame_classifier, POSE_LANDMARK_NAMES

class TestPoseProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PoseProcessor()

    @patch('Mediapipe_Class.cv2.VideoCapture')
    @patch('Mediapipe_Class.mp.Image')
    @patch('Mediapipe_Class.mp.tasks.vision.PoseLandmarker.create_from_options')
    def test_processframe_with_person(self, mock_create, mock_mp_image, mock_capture):
        # Mock camera capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())  # retval, frame
        mock_capture.return_value = mock_cap

        # Mock mediapipe image
        mock_image = MagicMock()
        mock_mp_image.return_value = mock_image

        # Mock landmarker
        mock_landmarker = MagicMock()
        mock_result = MagicMock()
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.3
        mock_landmark.z = 0.1
        mock_landmark.presence = 0.9
        mock_result.pose_landmarks = [mock_landmark] * 33  # 33 landmarks
        mock_landmarker.detect.return_value = mock_result
        mock_create.return_value.__enter__.return_value = mock_landmarker

        result = self.processor.processframe()
        self.assertIsInstance(result, dict)
        self.assertIn('NOSE', result)
        self.assertIn('x', result['NOSE'])
        self.assertIn('y', result['NOSE'])
        self.assertIn('z', result['NOSE'])
        self.assertIn('presence', result['NOSE'])

    @patch('Mediapipe_Class.cv2.VideoCapture')
    def test_processframe_no_person(self, mock_capture):
        # Mock camera capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        mock_capture.return_value = mock_cap

        # Mock landmarker with no results
        with patch('Mediapipe_Class.mp.tasks.vision.PoseLandmarker.create_from_options') as mock_create:
            mock_landmarker = MagicMock()
            mock_result = MagicMock()
            mock_result.pose_landmarks = None
            mock_landmarker.detect.return_value = mock_result
            mock_create.return_value.__enter__.return_value = mock_landmarker

            result = self.processor.processframe()
            self.assertEqual(result, {})

class TestFrameClassifier(unittest.TestCase):
    def setUp(self):
        with patch('Mediapipe_Class.PoseProcessor') as mock_processor:
            mock_instance = MagicMock()
            mock_instance.processframe.return_value = {
                name: {'x': 0.5, 'y': 0.3, 'z': 0.1, 'presence': 0.9}
                for name in POSE_LANDMARK_NAMES
            }
            mock_processor.return_value = mock_instance
            self.classifier = frame_classifier()

    def test_get_face(self):
        face = self.classifier.get_face()
        expected_face_landmarks = POSE_LANDMARK_NAMES[:11]
        for landmark in expected_face_landmarks:
            self.assertIn(landmark, face)

    def test_get_body(self):
        body = self.classifier.get_body()
        expected_body_landmarks = POSE_LANDMARK_NAMES[11:]
        for landmark in expected_body_landmarks:
            self.assertIn(landmark, body)

    def test_get_whole_person(self):
        whole = self.classifier.get_whole_person()
        self.assertEqual(len(whole), len(POSE_LANDMARK_NAMES))
        for landmark in POSE_LANDMARK_NAMES:
            self.assertIn(landmark, whole)

if __name__ == '__main__':
    unittest.main()