import boto3
import os

"""Rekognition face collection and matching wrapper."""


class FaceMatcher:
    def __init__(self, collection_id: str, bucket_name: str, confidence_threshold: float = 80.0):
        self.collection_id = collection_id
        self.bucket_name = bucket_name
        self.confidence_threshold = confidence_threshold
        self.rekognition = boto3.client("rekognition", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.rekognition.describe_collection(CollectionId=self.collection_id)
        except self.rekognition.exceptions.ResourceNotFoundException:
            self.rekognition.create_collection(CollectionId=self.collection_id)
            print(f"[Rekognition] Created collection: {self.collection_id}")

    def index_face(self, image_bytes: bytes, person_id: str) -> str:
        """Upload image to S3 and index in Rekognition collection."""
        # Store image in S3
        s3_key = f"faces/{person_id}.jpg"
        self.s3.put_object(Bucket=self.bucket_name, Key=s3_key, Body=image_bytes)

        # Index face in Rekognition
        response = self.rekognition.index_faces(
            CollectionId=self.collection_id,
            Image={"S3Object": {"Bucket": self.bucket_name, "Name": s3_key}},
            ExternalImageId=person_id,
            DetectionAttributes=["ALL"],
            MaxFaces=1,
        )

        if not response.get("FaceRecords"):
            raise ValueError(f"No face detected in image for {person_id}")

        face_id = response["FaceRecords"][0]["Face"]["FaceId"]
        print(f"[Rekognition] Indexed face for {person_id} → face_id: {face_id}")
        return face_id

    def match_face(self, image_bytes: bytes) -> str | None:
        """Search Rekognition collection for a matching face from camera frame."""
        try:
            response = self.rekognition.search_faces_by_image(
                CollectionId=self.collection_id,
                Image={"Bytes": image_bytes},
                MaxFaces=1,
                FaceMatchThreshold=self.confidence_threshold,
            )
            matches = response.get("FaceMatches", [])
            if matches:
                person_id = matches[0]["Face"]["ExternalImageId"]
                confidence = matches[0]["Similarity"]
                print(f"[Rekognition] Matched: {person_id} ({confidence:.1f}%)")
                return person_id
            return None
        except self.rekognition.exceptions.InvalidParameterException:
            # No face in frame
            return None
        except Exception as e:
            raise RuntimeError(f"Rekognition match error: {e}")
