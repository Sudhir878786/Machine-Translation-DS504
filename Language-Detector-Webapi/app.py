import argparse
from os import environ

from flask import Flask, escape, request, render_template

from models import LanguageDetector

language_detector = None

app = Flask(__name__)


@app.route("/api/v1/predict", methods=["POST"])
def predict_language():
    global language_detector

    json_body = request.get_json()

    if "input_text" not in json_body:
        return "Must have input text", 400

    text = json_body["input_text"]

    if len(text) == 0:
        return "Must have valid input text in request body", 400

    lang = language_detector.predict(text)
    print(f"Text={text} Lang={lang}")

    return lang, 200


@app.route("/api/v1/status", methods=["GET"])
def get_status():
    return "ok", 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Language Detector Web Api")
    language_detector = LanguageDetector(
        "../Language-Detector/models/Hansard-Multi30k/vocab.gz",
        "../Language-Detector/models/Hansard-Multi30k/model.pt",
    )

    app.run(host="0.0.0.0", port=environ.get("PORT", 5003))

