import boto3

"""List all records in the interactions DynamoDB table for debugging."""

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('interactions')
result = table.scan()
print(f'Total records: {len(result["Items"])}')
for item in result['Items']:
    print(f"  person: {item['person_id']} | date: {item['timestamp'][:10]} | text: {item['description'][:60]}")