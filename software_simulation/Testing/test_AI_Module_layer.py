import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the software_simulation directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AI_Module_layer import AI_LAYER, poses, Person_status

class TestAILayer(unittest.TestCase):
    def setUp(self):
        self.ai_layer = AI_LAYER()

    def test_get_elder_pose(self):
        pose_sequence = [
            {'LEFT_SHOULDER': {'x': 0.4, 'y': 0.4, 'z': 0.0},
             'RIGHT_SHOULDER': {'x': 0.6, 'y': 0.41, 'z': 0.0},
             'LEFT_HIP': {'x': 0.45, 'y': 0.55, 'z': 0.0},
             'RIGHT_HIP': {'x': 0.55, 'y': 0.56, 'z': 0.0},
             'LEFT_KNEE': {'x': 0.44, 'y': 0.8, 'z': 0.0},
             'RIGHT_KNEE': {'x': 0.56, 'y': 0.79, 'z': 0.0}}
        ]
        features = {
            'is_horizontal': False,
            'movement_score': 0.01,
            'recent_drop': 0.0,
            'shoulder_hip_height': 0.25,
            'vertical_span': 0.35,
            'depth_spread': 0.01
        }

        with patch.object(self.ai_layer.frame_classifier_instance, 'get_sequence', return_value=pose_sequence):
            with patch.object(self.ai_layer.frame_classifier_instance, 'compute_spatial_features', return_value=features):
                result = self.ai_layer.Get_Elder_Pose()

        self.assertEqual(result, 'STANDING')
        self.assertEqual(self.ai_layer.person_pose, 'STANDING')
        self.assertEqual(self.ai_layer.old_pose, 'STANDING')

    @patch('AI_Module_layer.ollama.generate')
    def test_get_elder_status(self, mock_generate):
        # Mock the ollama response
        mock_response = MagicMock()
        mock_response.response = 'Normal'
        mock_generate.return_value = mock_response

        sensor_data = [{'heart_rate': 70, 'temperature': 36.5}]
        result = self.ai_layer.Get_Elder_status(sensor_data)
        self.assertEqual(result, 'Normal')

    def test_prompt_builder(self):
        with patch.object(self.ai_layer.frame_classifier_instance, 'get_whole_person', return_value={}):
            prompt = self.ai_layer.Prompt_builder()
        self.assertIn('Body landmarks:', prompt)
        self.assertIn('Respond with one word:', prompt)

    def test_get_stroke_risk(self):
        # Mock the frame classifier and stroke detector
        with patch.object(self.ai_layer.frame_classifier_instance, 'get_whole_person', return_value={}) as mock_get:
            with patch.object(self.ai_layer.stroke_detector, 'get_stroke_risk', return_value={'risk_level': 'LOW'}) as mock_stroke:
                result = self.ai_layer.Get_Stroke_Risk()
                self.assertEqual(result['risk_level'], 'LOW')
                mock_get.assert_called_once()
                mock_stroke.assert_called_once_with({})

if __name__ == '__main__':
    unittest.main()