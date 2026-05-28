"""Emotion detection logic."""

from __future__ import annotations

from typing import Any

import requests

API_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NLP/text/emotion"
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
POSITIVE_WORDS = {"happy", "joy", "love", "great", "excellent", "thrilled", "good"}
NEGATIVE_WORDS = {"sad", "angry", "mad", "terrible", "bad", "fear", "disgust"}


def _empty_response(dominant_emotion: str | None = None) -> dict[str, Any]:
    """Return a predictable default response."""

    result: dict[str, Any] = {emotion: None for emotion in EMOTIONS}
    result["dominant_emotion"] = dominant_emotion
    return result


def _local_emotion_response(text_to_analyze: str) -> dict[str, Any]:
    """Provide a basic local fallback when the remote service is unavailable."""

    lowered = text_to_analyze.lower()
    result = {emotion: 0.0 for emotion in EMOTIONS}

    if any(word in lowered for word in POSITIVE_WORDS):
        result["joy"] = 0.9
        result["sadness"] = 0.05
        result["anger"] = 0.03
        result["fear"] = 0.01
        result["disgust"] = 0.01
    elif any(word in lowered for word in NEGATIVE_WORDS):
        result["sadness"] = 0.6
        result["anger"] = 0.2
        result["fear"] = 0.1
        result["disgust"] = 0.1
        result["joy"] = 0.0
    else:
        result["joy"] = 0.2
        result["fear"] = 0.2
        result["sadness"] = 0.2
        result["anger"] = 0.2
        result["disgust"] = 0.2

    result["dominant_emotion"] = max(EMOTIONS, key=result.get)
    return result

def emotion_detector(text_to_analyze: str) -> dict[str, Any]:
    """Analyze text and return a normalized emotion result."""

    if not text_to_analyze or not text_to_analyze.strip():
        return _empty_response("invalid text")

    payload = {"text": text_to_analyze}
    try:
        response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=10)
    except requests.RequestException:
        return _local_emotion_response(text_to_analyze)

    if response.status_code == 400:
        return _empty_response("invalid text")

    if response.status_code != 200:
        return _local_emotion_response(text_to_analyze)

    response_json = response.json()
    emotion_data = response_json["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotion_data, key=emotion_data.get)

    result = {emotion: emotion_data[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = dominant_emotion
    return result
