"""Unit tests for emotion detection."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Test the emotion detector behavior."""

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_happy_path(self, mock_post: Mock) -> None:
        """The detector should return normalized emotion data."""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [
                {
                    "emotion": {
                        "anger": 0.1,
                        "disgust": 0.05,
                        "fear": 0.02,
                        "joy": 0.8,
                        "sadness": 0.03,
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = emotion_detector("I am very happy today")

        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(result["joy"], 0.8)
        self.assertEqual(result["anger"], 0.1)

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_invalid_text(self, mock_post: Mock) -> None:
        """Status code 400 should produce the invalid text response."""

        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = emotion_detector("   ")

        self.assertEqual(result["dominant_emotion"], "invalid text")
        self.assertIsNone(result["anger"])
        self.assertIsNone(result["joy"])

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_emotion_detector_unknown_error(self, mock_post: Mock) -> None:
        """Unexpected errors should still return a stable response shape."""

        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = emotion_detector("This should still return a dict")

        self.assertEqual(result["dominant_emotion"], "anger")
        self.assertEqual(result["anger"], 0.2)
        self.assertIn("sadness", result)


if __name__ == "__main__":
    unittest.main()
