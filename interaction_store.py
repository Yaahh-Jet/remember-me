import boto3
import uuid
import os
from dataclasses import dataclass
from datetime import datetime, timezone

"""DynamoDB storage helper for user interaction history."""


@dataclass
class InteractionRecord:
    record_id: str
    person_id: str
    description: str
    timestamp: str


class InteractionStore:
    def __init__(self, table_name: str):
        self.table_name = table_name
        dynamodb = boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.table = dynamodb.Table(table_name)

    def add_interaction(self, person_id: str, description: str) -> InteractionRecord:
        """Persist an interaction event for a person and return metadata."""
        record = InteractionRecord(
            record_id=str(uuid.uuid4()),
            person_id=person_id,
            description=description,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self.table.put_item(Item={
                "record_id":   record.record_id,
                "person_id":   record.person_id,
                "description": record.description,
                "timestamp":   record.timestamp,
            })
            print(f"[DynamoDB] Stored interaction for {person_id}")
            return record
        except Exception as e:
            raise RuntimeError(f"DynamoDB write error: {e}")

    def get_interactions(self, person_id: str) -> list[InteractionRecord]:
        """Query last 10 interactions for person by timestamp descending."""
        try:
            response = self.table.query(
                IndexName="person_id-timestamp-index",
                KeyConditionExpression=boto3.dynamodb.conditions.Key("person_id").eq(person_id),
                ScanIndexForward=False,  # newest first
                Limit=10,
            )
            records = [
                InteractionRecord(
                    record_id=item["record_id"],
                    person_id=item["person_id"],
                    description=item["description"],
                    timestamp=item["timestamp"],
                )
                for item in response.get("Items", [])
            ]
            print(f"[DynamoDB] Fetched {len(records)} interactions for {person_id}")
            return records
        except Exception as e:
            raise RuntimeError(f"DynamoDB query error: {e}")
