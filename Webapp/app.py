from flask import Flask, escape, request, render_template
import requests
import aiohttp, asyncio

from aiohttp import ClientSession

import argparse
from os import environ

english_to_french_translator = None
french_to_english_translator = None
language_detector = None

app = Flask(__name__)


class TranslatorWebService:
    def __init__(self, url):
        ''' Initialize the web service '''
        self.url = url

    async def get_service_status_async(self, session):
        url = self.url + "/api/v1/status"
        is_successful = False

        async with session.get(url) as response:
            print(f"Trying to get status of ({url})")
            try:
                response = await session.request(method="GET", url=url)
                response.raise_for_status()
                print(f"Response status ({url}): {response.status}")
                is_successful = True

            except Exception as error:
                print(f"An error ocurred while trying {url}: {error}")

        return is_successful

    def translate_text(self, text):
        ''' Translate text from a web service

            Parameters
            ----------
            text : str
                The text to translate

            Returns
            -------
            translated_text : str
                The translated text
        '''

        url = self.url + "/api/v1/translate"
        response = requests.post(url, json={"input_text": text})
        response.raise_for_status()

        translated_text = response.text
        return translated_text


class LanguageDetectorWebService:
    def __init__(self, url):
        ''' Initialize the web service '''
        self.url = url

    async def get_service_status_async(self, session):
        url = self.url + "/api/v1/status"
        is_successful = False

        try:
            async with session.get(url) as response:
                print(f"Trying to get status of ({url})")
                try:
                    response = await session.request(method="GET", url=url)
                    print(response)
                    response.raise_for_status()
                    print(f"Response status ({url}): {response.status}")
                    is_successful = True

                except Exception as error:
                    print(f"An error ocurred while trying {url}: {error}")
        except Exception as error:
            print(f"An error ocurred while trying {url}: {error}")

        return is_successful

    def predict_language(self, text: str):
        ''' Predicts the language of text by calling a web service

            Parameters
            ----------
            text : str
                The text with the unknown langauge
            
            Returns
            -------
            language : str
                The predicted langauge of the text {'en', 'fr'}
        '''
        url = self.url + "/api/v1/predict"
        response = requests.post(url, json={"input_text": text})
        response.raise_for_status()

        language = response.text
        return language


def get_status_of_all_services():
    global english_to_french_translator, french_to_english_translator, language_detector

    # Check if the translator services are online
    async def get_status_async():
        async with aiohttp.ClientSession() as session:
            future_responses = [
                english_to_french_translator.get_service_status_async(session),
                french_to_english_translator.get_service_status_async(session),
                language_detector.get_service_status_async(session),
            ]
            responses = await asyncio.gather(*future_responses, return_exceptions=True)
            return responses

    return asyncio.run(get_status_async())


@app.route("/api/v1/translate", methods=["POST"])
def get_translation():
    global english_to_french_translator, french_to_english_translator

    json_body = request.get_json()

    if "input_text" not in json_body:
        return "Must have input text", 400

    input_text = json_body["input_text"]
    source_lang = json_body["source_lang"]
    target_lang = json_body["target_lang"]

    if source_lang == target_lang:
        return input_text, 200

    elif source_lang == "en" and target_lang == "fr":
        return english_to_french_translator.translate_text(input_text), 200

    elif source_lang == "fr" and target_lang == "en":
        return french_to_english_translator.translate_text(input_text), 200

    else:
        return "Illegal!", 400


@app.route("/api/v1/predict", methods=["POST"])
def get_language():
    global language_detector

    json_body = request.get_json()

    if "input_text" not in json_body:
        return "Must have input text", 400

    return language_detector.predict_language(json_body["input_text"]), 200


@app.route("/api/v1/status", methods=["GET"])
def get_status():
    responses = get_status_of_all_services()
    print(responses)

    if all(responses):
        return "ok", 200
    else:
        return "fail", 500


@app.route("/")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web app")
    parser.add_argument("--port", metavar="p", type=int, default=5000, help="Port")
    opts = parser.parse_args()

    # Set up the services
    english_to_french_translator = TranslatorWebService(
        environ.get("EN_FR_TRANSLATOR_ENDPOINT", "http://localhost:5001")
    )
    french_to_english_translator = TranslatorWebService(
        environ.get("FR_EN_TRANSLATOR_ENDPOINT", "http://localhost:5002")
    )
    language_detector = LanguageDetectorWebService(
        environ.get("LANGUAGE_DETECTOR_ENDPOINT", "http://localhost:5003")
    )

    # Start the web service
    app.run(host="0.0.0.0", port=environ.get("PORT", 5000), debug=environ.get("DEBUG", "False") == "True")
