import boto3
import os
import time
import uuid
import wave
import threading
import pyaudio
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")

# Audio config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class TranscribeHandler:
    def __init__(self):
        self.s3 = boto3.client("s3", region_name=REGION)
        self.transcribe = boto3.client("transcribe", region_name=REGION)
        self.bucket = os.getenv("S3_BUCKET", "memoire-faces-yjt")

        self.recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None

    def start_recording(self):
        """Start capturing mic audio in background thread."""
        self.recording = True
        self.frames = []
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[Transcribe] Recording started...")

    def _record_loop(self):
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        while self.recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(data)

        self.stream.stop_stream()
        self.stream.close()

    def stop_and_transcribe(self) -> str:
        """Stop recording, upload to S3, run Transcribe job, return transcript."""
        self.recording = False
        if self.thread:
            self.thread.join(timeout=2)

        if not self.frames:
            return ""

        # Save WAV locally
        job_id = str(uuid.uuid4())[:8]
        wav_path = f"recording_{job_id}.wav"
        s3_key = f"recordings/{job_id}.wav"

        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))

        print(f"[Transcribe] Saved recording: {wav_path}")

        # Upload to S3
        self.s3.upload_file(wav_path, self.bucket, s3_key)
        os.remove(wav_path)
        print(f"[Transcribe] Uploaded to S3: {s3_key}")

        # Start Transcribe job
        job_name = f"memoire-{job_id}"
        self.transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": f"https://s3.amazonaws.com/{self.bucket}/{s3_key}"},
            MediaFormat="wav",
            LanguageCode="en-US",
        )
        print(f"[Transcribe] Job started: {job_name}")

        # Poll until complete
        transcript = self._poll_job(job_name)

        # Cleanup S3 recording
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
        except Exception:
            pass

        return transcript

    def _poll_job(self, job_name: str, timeout: int = 120) -> str:
        """Poll Transcribe job until done, return transcript text."""
        start = time.time()
        while time.time() - start < timeout:
            response = self.transcribe.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                import urllib.request, json as _json
                with urllib.request.urlopen(uri) as r:
                    data = _json.loads(r.read())
                transcript = data["results"]["transcripts"][0]["transcript"]
                print(f"[Transcribe] Done: {transcript[:80]}...")
                return transcript

            elif status == "FAILED":
                print("[Transcribe] Job failed.")
                return ""

            print(f"[Transcribe] Status: {status} — waiting...")
            time.sleep(5)

        print("[Transcribe] Timed out.")
        return ""
