import boto3
import os
import tempfile
import subprocess

"""Text-to-speech support using AWS Polly."""


def speak_summary(text: str):
    """Generate speech wav for given text and play locally (Windows)."""

    try:
        polly = boto3.client("polly", region_name=os.getenv("AWS_REGION", "us-east-1"))
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat="pcm",        # ← changed to pcm
            VoiceId="Joanna",
            Engine="neural",
            SampleRate="16000",
        )

        # write raw pcm to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            # add WAV header manually so winsound can play binary PCM
            import wave, struct
            pcm_data = response["AudioStream"].read()
            f_wav = wave.open(f.name, 'wb')
            f_wav.setnchannels(1)
            f_wav.setsampwidth(2)
            f_wav.setframerate(16000)
            f_wav.writeframes(pcm_data)
            f_wav.close()
            tmp_path = f.name

        # play WAV on Windows
        import winsound
        winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
        print(f"[Polly] Speaking summary")

    except Exception as e:
        print(f"[Polly] Error: {e}")