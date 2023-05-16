import boto3
import os
from neural_network_model.model import SETTING

# AWS credentials
S3_AWS_ACCESS_KEY = os.getenv("S3_AWS_ACCESS_KEY")
S3_AWS_SECRET_ACCESS_KEY = os.getenv("S3_AWS_SECRET_ACCESS_KEY")
print(S3_AWS_ACCESS_KEY)
print(S3_AWS_SECRET_ACCESS_KEY)

# access a bitbucket and download the whole directory
s3 = boto3.client(
    "s3",
    aws_access_key_id=S3_AWS_ACCESS_KEY,
    aws_secret_access_key=S3_AWS_SECRET_ACCESS_KEY,
)
s3.download_file(
    Bucket=SETTING.S3_BUCKET_SETTING.BUCKET_NAME,
    Key=SETTING.S3_BUCKET_SETTING.BUCKET_KEY,
    Filename=SETTING.S3_BUCKET_SETTING.BUCKET_FILE_NAME,
)

