import boto3
import os
import pathlib
from neural_network_model.model import SETTING
# AWS credentials
ACCESS_KEY = 'AKIA33OJLZR72SOCJHH6'
SECRET_KEY = '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'

# S3 bucket details
BUCKET_NAME = SETTING.S3_BUCKET_SETTING.BUCKET_NAME
FOLDER_NAME = SETTING.S3_BUCKET_SETTING.BUCKET_MAIN_FOLDER_NAME
# Specify the desired download location here
DOWNLOAD_LOCATION = SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION
# check if the download location exists, if not create it
if not os.path.exists(DOWNLOAD_LOCATION):
    pathlib.Path(DOWNLOAD_LOCATION).mkdir(parents=True, exist_ok=True)

# Connect to AWS
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)
s3 = session.client('s3')

# Function to recursively retrieve all objects within a folder
def get_all_objects(bucket, prefix):
    objects = []
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    contents = response.get('Contents', [])
    objects.extend(contents)

    while response.get('NextContinuationToken'):
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=response['NextContinuationToken'])
        contents = response.get('Contents', [])
        objects.extend(contents)

    return objects


# Retrieve all objects within the folder (including subfolders)
objects = get_all_objects(BUCKET_NAME, FOLDER_NAME)

# Download each object
for obj in objects:
    key = obj['Key']
    if obj['Size'] > 0:  # Exclude folders (size = 0)
        filename = os.path.basename(key)  # Extract the filename from the object key
        file_path = os.path.join(DOWNLOAD_LOCATION, filename)  # Construct the download file path
        s3.download_file(BUCKET_NAME, key, file_path)
        print(f'Downloaded: {file_path}')

print('All files downloaded successfully')
