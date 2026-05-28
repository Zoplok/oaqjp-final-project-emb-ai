"""Flask web server for emotion detection."""

from __future__ import annotations

from flask import Flask, render_template, request

from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """Render the home page."""

    return render_template("index.html")


@app.route("/emotionDetector", methods=["GET"])
def detect_emotion() -> str:
    """Return formatted emotion detection output."""

    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again!"

    result = emotion_detector(text_to_analyze)
    if result["dominant_emotion"] == "invalid text":
        return "Invalid text! Please try again!"

    response = (
        "For the given statement, the system response is "
        f"'{result['dominant_emotion']}'. "
        "The following were the emotions detected: "
        f"anger: {result['anger']}, "
        f"disgust: {result['disgust']}, "
        f"fear: {result['fear']}, "
        f"joy: {result['joy']}, "
        f"sadness: {result['sadness']}."
    )
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
