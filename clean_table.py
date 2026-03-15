import boto3

"""Delete demo DynamoDB table and wait for completion."""

client = boto3.client("dynamodb", region_name="us-east-1")
client.delete_table(TableName="interactions")
print("Deleted. Waiting...")
waiter = client.get_waiter("table_not_exists")
waiter.wait(TableName="interactions")
print("Done. Now run create_table.py")