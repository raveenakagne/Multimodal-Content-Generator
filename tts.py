from deepgram import DeepgramClient, SpeakOptions

DEEPGRAM_API_KEY = "1381133e0c0ee8b7329e039e4754ffb225ff4c79"

TEXT = {
    "text": "Deepgram is great for real-time conversations… and also, you can build apps for things like customer support, logistics, and more. What do you think of the voices?"
}
FILENAME = "tts_audio.mp3"


def main():
    try:
        deepgram = DeepgramClient(DEEPGRAM_API_KEY)

        options = SpeakOptions(
            model="aura-asteria-en",
        )

        response = deepgram.speak.v("1").save(FILENAME, TEXT, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    main()