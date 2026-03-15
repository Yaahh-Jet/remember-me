import os
import wave
import threading
import pyaudio
import speech_recognition as sr
from dotenv import load_dotenv

"""Handles audio capture and speech recognition.

Records from microphone in raw form, provides live interim text through callback,
and performs full final transcript on saved WAV file.
"""

load_dotenv()

CHUNK    = 1024
FORMAT   = pyaudio.paInt16
CHANNELS = 1
RATE     = 16000


class TranscribeHandler:
    def __init__(self):
        self.recording         = False
        self.frames            = []
        self.audio             = pyaudio.PyAudio()
        self.stream            = None
        self.thread            = None
        self.on_interim_text   = None  # callback for live display
        self.recognizer        = sr.Recognizer()
        self.live_thread       = None

    def start_recording(self):
        """Begin recording and triggering live transcription in parallel."""
        self.recording = True
        self.frames    = []

        # main recording thread
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

        # live recognition thread
        self.live_thread = threading.Thread(target=self._live_loop, daemon=True)
        self.live_thread.start()
        print("[Transcribe] Recording started...")

    def _record_loop(self):
        """Capture raw audio frames for final transcription."""
        self.stream = self.audio.open(
            format=FORMAT, channels=CHANNELS,
            rate=RATE, input=True,
            frames_per_buffer=CHUNK,
        )
        while self.recording:
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            self.frames.append(data)
        self.stream.stop_stream()
        self.stream.close()

    def _live_loop(self):
        """Show live interim text using mic while recording."""
        mic = sr.Microphone()
        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
        while self.recording:
            try:
                with mic as source:
                    audio = self.recognizer.listen(
                        source, timeout=2, phrase_time_limit=6
                    )
                text = self.recognizer.recognize_google(audio)
                if text and self.on_interim_text:
                    self.on_interim_text(f"🎙 {text}")
                    print(f"[Live] {text}")
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except Exception as e:
                pass

    def stop_and_transcribe(self) -> str:
        """Stop recording and transcribe full audio."""
        self.recording = False

        if self.thread:
            self.thread.join(timeout=3)

        if not self.frames:
            print("[Transcribe] No audio captured.")
            return ""

        # save WAV
        wav_path = "recording_final.wav"
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))

        print(f"[Transcribe] Saved final recording: {wav_path}")

        # transcribe full recording
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
            transcript = self.recognizer.recognize_google(audio_data)
            print(f"[Transcribe] Final transcript: {transcript}")
            os.remove(wav_path)
            return transcript
        except sr.UnknownValueError:
            print("[Transcribe] Could not understand audio.")
            return ""
        except Exception as e:
            print(f"[Transcribe] Error: {e}")
            return ""

