from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage, firestore

# Initialize clients
storage_client = storage.Client()
firestore_client = firestore.Client()
speech_client = speech.SpeechClient()

def process_audio(query_id):
    # Get raw data path from Firestore
    doc_ref = firestore_client.collection("queries").document(query_id)
    doc = doc_ref.get()
    if not doc.exists:
        print(f"No document found for query_id: {query_id}")
        return

    data = doc.to_dict()
    raw_data_path = data["raw_data_path"]

    # GCS URI for the audio file
    gcs_uri = f"gs://{raw_data_path}"

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
    )

    response = speech_client.recognize(config=config, audio=audio)

    # Concatenate transcription results
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    # Update Firestore with transcript
    doc_ref.update({
        "metadata.transcript": transcript
    })
