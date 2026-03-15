import boto3

"""Create DynamoDB interactions table used by MemoireAR demo."""

dynamodb = boto3.client("dynamodb", region_name="us-east-1")

dynamodb.create_table(
    TableName="interactions",
    AttributeDefinitions=[
        {"AttributeName": "record_id",  "AttributeType": "S"},
        {"AttributeName": "person_id",  "AttributeType": "S"},
        {"AttributeName": "timestamp",  "AttributeType": "S"},
    ],
    KeySchema=[
        {"AttributeName": "record_id", "KeyType": "HASH"},
    ],
    GlobalSecondaryIndexes=[
        {
            "IndexName": "person_id-timestamp-index",
            "KeySchema": [
                {"AttributeName": "person_id",  "KeyType": "HASH"},
                {"AttributeName": "timestamp",  "KeyType": "RANGE"},
            ],
            "Projection": {"ProjectionType": "ALL"},
        }
    ],
    BillingMode="PAY_PER_REQUEST",
)
print("Table created! Waiting...")
waiter = dynamodb.get_waiter("table_exists")
waiter.wait(TableName="interactions")
print("Table is ACTIVE — done!")