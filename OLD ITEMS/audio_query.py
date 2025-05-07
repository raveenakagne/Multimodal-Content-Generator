import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime, timezone
import os
from google.cloud import storage
from google.oauth2 import service_account

# Set up Google Cloud credentials
client = storage.Client()

def record_audio(duration=5, filename='recorded_audio.wav', fs=44100):
    """
    Records audio from the microphone and saves it to a WAV file.

    Args:
        duration (int): Duration of the recording in seconds.
        filename (str): The filename to save the recording.
        fs (int): Sampling frequency (samples per second).
    """
    print(f"Recording for {duration} seconds...")
    # Record audio for the given duration
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save the recording to a WAV file
    write(filename, fs, recording)
    print(f"Audio saved to {filename}.")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the GCP bucket.

    Args:
        bucket_name (str): The name of the GCP bucket.
        source_file_name (str): The local path of the file to upload.
        destination_blob_name (str): The name of the file in the bucket.
    """
    try:
        # Specify the bucket
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(source_file_name)
        print(f"File {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")

    except Exception as e:
        print(f"Failed to upload {source_file_name} to GCP bucket {bucket_name}. Error: {e}")

if __name__ == '__main__':
    # Prompt user for duration and filename
    duration = int(input("Enter recording duration in seconds: "))
    filename = input("Enter filename to save recording (e.g., 'recorded_audio.wav'): ") or 'recorded_audio.wav'

    # Record the audio
    record_audio(duration=duration, filename=filename)

    # GCP bucket details
    bucket_name = "audio-query"
    
    # Generate a timezone-aware timestamped filename for GCP storage
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_filename = os.path.splitext(filename)[0]
    destination_blob_name = f"{base_filename}-{timestamp}.wav"

    # Upload the audio file to GCP with the timestamped filename
    upload_to_gcs(bucket_name, filename, destination_blob_name)
