# Requirements Document

## Introduction

The Dementia Face Recognition Assistant is a terminal-based application that helps dementia patients identify familiar people using their webcam. When a face is detected, the system matches it against a known collection, retrieves past interaction history, and presents a warm, plain-language summary to the patient. The goal is to reduce confusion and anxiety by giving patients immediate, friendly context about who they are seeing.

## Glossary

- **System**: The Dementia Face Recognition Assistant application
- **Patient**: The dementia patient using the application
- **Webcam_Capture**: The component responsible for capturing images from the connected webcam
- **Face_Matcher**: The component that interfaces with Amazon Rekognition to index and match faces
- **Interaction_Store**: The component that interfaces with Amazon DynamoDB to store and retrieve interaction history
- **Summary_Generator**: The component that interfaces with the Anthropic API (Claude Haiku) to produce plain-language summaries
- **Face_Collection**: The Amazon Rekognition collection containing indexed face images of known persons
- **Interaction_Record**: A DynamoDB item representing a past interaction between the patient and a known person
- **Known_Person**: A person whose face has been indexed in the Face_Collection and who has at least one Interaction_Record
- **Match_Confidence**: The similarity score (0–100) returned by Amazon Rekognition for a face comparison
- **Confidence_Threshold**: The minimum Match_Confidence value (default: 80) required to consider a face match valid
- **S3_Store**: The Amazon S3 bucket used to store face images
- **Demo_Loader**: The setup_demo.py script that preloads demo faces and interactions

---

## Requirements

### Requirement 1: Webcam Image Capture

**User Story:** As a patient, I want the system to capture my view from the webcam, so that it can identify the person in front of me.

#### Acceptance Criteria

1. WHEN the application starts, THE Webcam_Capture SHALL open the default connected webcam device.
2. WHEN the webcam is opened, THE Webcam_Capture SHALL capture a single still image frame.
3. IF no webcam device is detected, THEN THE Webcam_Capture SHALL display a descriptive error message in the terminal and exit gracefully.
4. IF the webcam fails to capture a frame, THEN THE Webcam_Capture SHALL display a descriptive error message in the terminal and exit gracefully.
5. THE Webcam_Capture SHALL encode the captured frame as a JPEG image in memory before passing it to the Face_Matcher.

---

### Requirement 2: Face Indexing

**User Story:** As a caregiver, I want to register known persons' faces into the system, so that the patient can be shown information about people they know.

#### Acceptance Criteria

1. WHEN a face image and a person identifier are provided, THE Face_Matcher SHALL index the face into the Face_Collection using Amazon Rekognition.
2. WHEN indexing succeeds, THE Face_Matcher SHALL store the source face image in the S3_Store.
3. IF Amazon Rekognition detects no face in the provided image, THEN THE Face_Matcher SHALL return a descriptive error indicating no face was found.
4. IF the Amazon Rekognition API call fails, THEN THE Face_Matcher SHALL raise an exception with the API error detail.
5. THE Face_Matcher SHALL associate the Rekognition-assigned face ID with the person identifier for later retrieval.

---

### Requirement 3: Face Matching

**User Story:** As a patient, I want the system to identify who is in front of me, so that I can be reminded of who that person is.

#### Acceptance Criteria

1. WHEN a captured image is provided, THE Face_Matcher SHALL submit it to Amazon Rekognition to search the Face_Collection.
2. WHEN Amazon Rekognition returns one or more matches above the Confidence_Threshold, THE Face_Matcher SHALL return the person identifier associated with the highest-confidence match.
3. WHEN Amazon Rekognition returns no matches above the Confidence_Threshold, THE Face_Matcher SHALL return a result indicating the person is unrecognized.
4. IF the Amazon Rekognition API call fails, THEN THE Face_Matcher SHALL raise an exception with the API error detail.
5. THE Face_Matcher SHALL use a Confidence_Threshold of 80 by default, and SHALL accept an override value at initialization.

---

### Requirement 4: Interaction History Storage

**User Story:** As a caregiver, I want to record past interactions with the patient, so that the system can surface relevant memories.

#### Acceptance Criteria

1. WHEN a person identifier and interaction details are provided, THE Interaction_Store SHALL write an Interaction_Record to the DynamoDB table.
2. THE Interaction_Store SHALL record the person identifier, a human-readable description of the interaction, and an ISO 8601 timestamp on each Interaction_Record.
3. IF the DynamoDB write operation fails, THEN THE Interaction_Store SHALL raise an exception with the error detail.
4. THE Interaction_Store SHALL assign a unique identifier to each Interaction_Record at write time.

---

### Requirement 5: Interaction History Retrieval

**User Story:** As a patient, I want the system to fetch memories of my past interactions with a recognized person, so that I can be reminded of our relationship.

#### Acceptance Criteria

1. WHEN a person identifier is provided, THE Interaction_Store SHALL query the DynamoDB table and return all Interaction_Records associated with that person.
2. WHEN no Interaction_Records exist for the given person identifier, THE Interaction_Store SHALL return an empty list.
3. IF the DynamoDB query operation fails, THEN THE Interaction_Store SHALL raise an exception with the error detail.
4. THE Interaction_Store SHALL return Interaction_Records sorted by timestamp in descending order (most recent first).

---

### Requirement 6: Plain-Language Summary Generation

**User Story:** As a patient, I want to receive a warm, easy-to-understand summary about the recognized person, so that I feel reassured and oriented.

#### Acceptance Criteria

1. WHEN a person identifier and a list of Interaction_Records are provided, THE Summary_Generator SHALL send a prompt to Claude Haiku via the Anthropic API requesting a warm, plain-language summary.
2. WHEN the Anthropic API returns a response, THE Summary_Generator SHALL return the generated summary text.
3. WHEN the list of Interaction_Records is empty, THE Summary_Generator SHALL generate a summary indicating no past interactions are on record for that person.
4. IF the Anthropic API call fails, THEN THE Summary_Generator SHALL raise an exception with the API error detail.
5. THE Summary_Generator SHALL limit the generated summary to a maximum of 150 words to keep it readable for the patient.
6. THE Summary_Generator SHALL instruct Claude Haiku to use a warm, calm, and reassuring tone in the prompt.

---

### Requirement 7: Terminal Display

**User Story:** As a patient, I want to see the summary displayed clearly in the terminal, so that I can read it without confusion.

#### Acceptance Criteria

1. WHEN a summary is generated for a recognized person, THE System SHALL print the person's name and the summary text to the terminal.
2. WHEN the face is unrecognized, THE System SHALL print a calm, reassuring message to the terminal indicating the person was not recognized.
3. THE System SHALL visually separate the output from preceding terminal content using a blank line before and after the output block.

---

### Requirement 8: Demo Data Loading

**User Story:** As a developer or caregiver, I want to preload demo faces and interactions, so that the system can be demonstrated without manual setup.

#### Acceptance Criteria

1. WHEN the Demo_Loader is executed, THE Demo_Loader SHALL index at least two demo face images into the Face_Collection via the Face_Matcher.
2. WHEN the Demo_Loader is executed, THE Demo_Loader SHALL write at least two Interaction_Records per demo person into the Interaction_Store.
3. IF a demo face image file is not found at the expected path, THEN THE Demo_Loader SHALL print a descriptive error and skip that entry without halting execution.
4. WHEN the Demo_Loader completes, THE Demo_Loader SHALL print a confirmation message listing the persons successfully loaded.

---

### Requirement 9: End-to-End Orchestration

**User Story:** As a patient, I want to run a single command that handles everything automatically, so that I don't need to understand the underlying steps.

#### Acceptance Criteria

1. WHEN the application is launched via main.py, THE System SHALL execute the following steps in order: capture image, match face, fetch interaction history, generate summary, display summary.
2. IF any step raises an exception, THEN THE System SHALL catch the exception, print a descriptive error message to the terminal, and exit with a non-zero exit code.
3. THE System SHALL complete the full pipeline from image capture to summary display within 30 seconds under normal network conditions.
