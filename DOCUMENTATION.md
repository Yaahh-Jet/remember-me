# MemoireAR Project Documentation

## Overview
MemoireAR is a desktop demo app for dementia assistance. It uses face recognition to identify known individuals, records audio, transcribes speech, stores interactions in DynamoDB, generates a short memory summary via Bedrock, and optionally speaks it with AWS Polly.

## Main components
- `Mani.py`: Main GUI with overlay rendering, camera feed, face recognition, recording controls, and status updates. Recommended for newer UI.
- `main.py`: Alternate app UI with separate text widgets and similar logic.
- `transcribe_handler.py`: Handles mic recording with PyAudio, live interim transcription (SpeechRecognition + Google), and final transcription from saved WAV.
- `summary_generator.py`: Calls AWS Bedrock with an Anthropic model to produce a gentle summary message.
- `voice_output.py`: Uses AWS Polly to convert text to PCM, creates WAV file, and plays it on Windows with winsound.
- `face_handler.py`: Wraps Rekognition collection + faces matching.
- `interaction_store.py`: Wraps DynamoDB table operations for adding and fetching interactions.
- `setup_demo.py`: Seed demo data (photos and interactions) into Rekognition and DynamoDB.
- `test_audio.py`: Verify Polly TTS and playback.

## Required setup
1. Python (>=3.9) environment.
2. Install dependencies:
   ```bash
   pip install -r files/requirements.txt
   ```
3. Create `.env` with keys:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `REKOGNITION_COLLECTION` (e.g., `memoire-faces`)
   - `S3_BUCKET` (e.g., `memoire-faces-demo`)
   - `DYNAMODB_TABLE` (e.g., `interactions`)
4. Optional: Create DynamoDB table run `files/create_table.py` and then run `files/setup_demo.py`.

## Run
- Launch app:
  - `python files/Mani.py` (recommended)
  - `python files/main.py` (alternate)
- Speak into mic when recording is active.
- Confirm live transcription appears at bottom overlay.
- Stop recording to save interaction, generate summary, and optionally trigger TTS.

## Notes
- Fix `SpeechRecognition` and `PyAudio` driver issues on Windows with `pipwin install pyaudio`.
- TTS currently uses `winsound` (Windows-only). For Linux/macOS, swap in `simpleaudio` or `playsound`.
- Live translation is not implemented; add AWS Translate call in `_on_interim` or _process_transcript if needed.

## Troubleshooting
- If no voice/transcription: verify USB mic presence and Windows microphone permission.
- If face detection fails: confirm Rekognition collection contains indexed face images.
- If summary fails: check AWS Bedrock model access and region configuration.
