# Implementation Plan: Dementia Face Recognition Assistant

## Overview

Implement a linear Python pipeline: webcam capture → face matching (Rekognition) → interaction history (DynamoDB) → summary generation (Claude Haiku) → terminal display. A separate `setup_demo.py` script pre-populates demo data.

## Tasks

- [ ] 1. Project setup and environment configuration
  - Create project directory structure: `main.py`, `setup_demo.py`, `components/` package with `__init__.py`
  - Create `requirements.txt` with dependencies: `boto3`, `anthropic`, `opencv-python`, `python-dotenv`
  - Create `.env.example` documenting all required environment variables (`AWS_REGION`, `REKOGNITION_COLLECTION`, `S3_BUCKET`, `DYNAMODB_TABLE`, `ANTHROPIC_API_KEY`, `CONFIDENCE_THRESHOLD`)
  - _Requirements: 9.1_

- [ ] 2. Implement `WebcamCapture` component
  - [ ] 2.1 Implement `WebcamCapture` class in `components/webcam_capture.py`
    - Open default webcam device (`cv2.VideoCapture(0)`)
    - Capture a single frame and encode as JPEG bytes via `cv2.imencode`
    - Call `sys.exit` with a descriptive message if no device found or capture fails
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write unit tests for `WebcamCapture`
    - Test successful capture returns non-empty bytes
    - Test graceful exit when `VideoCapture.isOpened()` returns False
    - Test graceful exit when `read()` returns failure
    - _Requirements: 1.3, 1.4_

- [ ] 3. Implement `InteractionRecord` dataclass and `InteractionStore` component
  - [ ] 3.1 Implement `InteractionRecord` dataclass and `InteractionStore` class in `components/interaction_store.py`
    - Define `InteractionRecord` with fields: `record_id` (UUID), `person_id`, `description`, `timestamp` (ISO 8601)
    - Implement `add_interaction(person_id, description)` — writes to DynamoDB, raises `RuntimeError` on failure
    - Implement `get_interactions(person_id)` — queries GSI `person_id-timestamp-index` with `ScanIndexForward=False`, returns list sorted descending; returns `[]` if none found; raises `RuntimeError` on failure
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4_

  - [ ]* 3.2 Write property test for `InteractionRecord` round-trip consistency
    - **Property 1: Round-trip consistency** — any record written via `add_interaction` must appear in `get_interactions` with identical fields
    - **Validates: Requirements 4.1, 4.2, 5.1**

  - [ ]* 3.3 Write unit tests for `InteractionStore`
    - Test `add_interaction` raises `RuntimeError` on DynamoDB failure (mock boto3)
    - Test `get_interactions` returns empty list when no records exist
    - Test `get_interactions` returns records sorted by timestamp descending
    - Test `get_interactions` raises `RuntimeError` on DynamoDB failure
    - _Requirements: 4.3, 5.2, 5.3, 5.4_

- [ ] 4. Implement `FaceMatcher` component
  - [ ] 4.1 Implement `FaceMatcher` class in `components/face_matcher.py`
    - Constructor accepts `collection_id`, `bucket_name`, `confidence_threshold=80.0`
    - Implement `index_face(image_bytes, person_id)` — uploads image to S3 at `faces/{person_id}.jpg`, calls `IndexFaces` on Rekognition collection, stores `ExternalImageId=person_id`; raises `ValueError` if no face detected; raises `RuntimeError` on API failure; returns `face_id`
    - Implement `match_face(image_bytes)` — calls `SearchFacesByImage` on collection; returns `person_id` of highest-confidence match above threshold, or `None` if no match; raises `RuntimeError` on API failure
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 4.2 Write property test for `FaceMatcher` confidence threshold invariant
    - **Property 2: Threshold invariant** — `match_face` must never return a `person_id` when the highest Rekognition similarity score is below `confidence_threshold`
    - **Validates: Requirements 3.2, 3.3, 3.5**

  - [ ]* 4.3 Write unit tests for `FaceMatcher`
    - Test `index_face` raises `ValueError` when Rekognition returns no `FaceRecords`
    - Test `index_face` raises `RuntimeError` on Rekognition API exception
    - Test `match_face` returns `None` when no faces match above threshold
    - Test `match_face` returns correct `person_id` for highest-confidence match
    - Test `match_face` raises `RuntimeError` on Rekognition API exception
    - _Requirements: 2.3, 2.4, 3.2, 3.3, 3.4_

- [ ] 5. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement `SummaryGenerator` component
  - [ ] 6.1 Implement `SummaryGenerator` class in `components/summary_generator.py`
    - Constructor accepts `api_key` and `max_words=150`
    - Implement `generate(person_id, interactions)` — builds a prompt instructing Claude Haiku to use a warm, calm, reassuring tone and stay within 150 words; sends via Anthropic messages API (`claude-haiku-*` model); returns response text; raises `RuntimeError` on API failure
    - When `interactions` is empty, prompt must instruct Claude to indicate no past interactions are on record
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [ ]* 6.2 Write property test for `SummaryGenerator` word-count invariant
    - **Property 3: Word-count invariant** — the generated summary must never exceed 150 words for any non-empty or empty interactions list
    - **Validates: Requirements 6.5**

  - [ ]* 6.3 Write unit tests for `SummaryGenerator`
    - Test `generate` raises `RuntimeError` on Anthropic API failure (mock client)
    - Test `generate` with empty interactions list produces a response (mock API)
    - _Requirements: 6.3, 6.4_

- [ ] 7. Implement `display_result` and wire `main.py` pipeline
  - [ ] 7.1 Implement `display_result` function in `main.py`
    - Print blank line, then person name + summary if recognized, or calm unrecognized message if `person_id` is `None`, then blank line
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 7.2 Implement `main()` in `main.py`
    - Load environment variables (via `python-dotenv` or `os.environ`)
    - Instantiate all components with values from environment
    - Execute pipeline in order: `WebcamCapture.capture()` → `FaceMatcher.match_face()` → `InteractionStore.get_interactions()` → `SummaryGenerator.generate()` → `display_result()`
    - Wrap entire pipeline in `try/except`; on any exception print descriptive error and call `sys.exit(1)`
    - _Requirements: 9.1, 9.2_

  - [ ]* 7.3 Write unit tests for `display_result`
    - Test recognized person output includes name and summary with surrounding blank lines
    - Test unrecognized person output prints calm message with surrounding blank lines
    - _Requirements: 7.1, 7.2, 7.3_

- [ ] 8. Implement `setup_demo.py` demo loader
  - [ ] 8.1 Implement `load_demo` function in `setup_demo.py`
    - Define at least two demo persons with local image paths and at least two interaction descriptions each
    - For each demo person: call `FaceMatcher.index_face()`; if image file not found, print descriptive warning and skip without halting
    - For each demo person: call `InteractionStore.add_interaction()` for each interaction description
    - After processing all persons, print confirmation listing successfully loaded persons
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 8.2 Write unit tests for `load_demo`
    - Test missing image file is skipped with a printed warning and execution continues
    - Test confirmation message lists only successfully loaded persons
    - _Requirements: 8.3, 8.4_

- [ ] 9. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation before proceeding
- Property tests validate universal correctness invariants; unit tests cover specific examples and edge cases
- All AWS clients should be instantiated with `boto3.client(region_name=os.environ.get("AWS_REGION", "us-east-1"))`
- Mock boto3 and Anthropic clients in tests to avoid live API calls
