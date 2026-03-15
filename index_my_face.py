import boto3
import os
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET", "memoire-faces-yjt")
COLLECTION = os.getenv("REKOGNITION_COLLECTION", "memoire-faces")
REGION = os.getenv("AWS_REGION", "us-east-1")

rekognition = boto3.client("rekognition", region_name=REGION)

# ── EDIT THIS ──────────────────────────────
persons = [
    {"person_id": "Dad",  "s3_key": "yajat.jpeg"},   # exact filename in S3
    {"person_id": "Son", "s3_key": "viren.jpeg"}, # add more as needed
]
# ───────────────────────────────────────────

for person in persons:
    try:
        response = rekognition.index_faces(
            CollectionId=COLLECTION,
            Image={
                "S3Object": {
                    "Bucket": BUCKET,
                    "Name": person["s3_key"]   # exact key in S3
                }
            },
            ExternalImageId=person["person_id"],  # this is what gets returned on match
            DetectionAttributes=["ALL"],
            MaxFaces=1,
        )
        if response["FaceRecords"]:
            print(f"✅ Indexed: {person['person_id']}")
        else:
            print(f"⚠️  No face detected in {person['s3_key']} — retake photo")
    except Exception as e:
        print(f"❌ Error for {person['person_id']}: {e}")