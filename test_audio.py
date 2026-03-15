import boto3
import wave
import winsound
import tempfile
import os

"""Simple AWS Polly tts validation script. Runs one sample utterance and plays it."""

polly = boto3.client("polly", region_name="us-east-1")

response = polly.synthesize_speech(
    Text="Hello! This is Memoire. Your daughter Sarah is here.",
    OutputFormat="pcm",
    VoiceId="Joanna",
    Engine="neural",
    SampleRate="16000",
)

pcm_data = response["AudioStream"].read()
tmp_path = os.path.join(tempfile.gettempdir(), "test_audio.wav")

with wave.open(tmp_path, 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)
    f.writeframes(pcm_data)

print(f"Playing audio from {tmp_path}")
winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
print("Done!")