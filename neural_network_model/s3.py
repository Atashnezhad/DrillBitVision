import boto3
import os
import pathlib
from neural_network_model.model import SETTING

# AWS credentials
ACCESS_KEY = ''
SECRET_KEY = ''

# S3 bucket details
BUCKET_NAME = SETTING.S3_BUCKET_SETTING.BUCKET_NAME
FOLDER_NAME = SETTING.S3_BUCKET_SETTING.BUCKET_MAIN_FOLDER_NAME
SUBFOLDERS = SETTING.S3_BUCKET_SETTING.SUBFOLDER_NAME
# Specify the desired download location here
DOWNLOAD_LOCATION = SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION
# check if the download location exists, if not create it
if not os.path.exists(DOWNLOAD_LOCATION):
    pathlib.Path(DOWNLOAD_LOCATION).mkdir(parents=True, exist_ok=True)

# Set the AWS region
region = "us-east-2"

# Create an S3 client
client = boto3.client(
    "s3", region_name=region,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Get the name of the S3 bucket
bucket_name = BUCKET_NAME

# Get the names of the subfolders in the S3 bucket
subfolder_names = SUBFOLDERS

# Create a local directory to store the downloaded files
local_directory = DOWNLOAD_LOCATION

# Iterate over the subfolders in the S3 bucket
for subfolder_name in subfolder_names:

    # Get the objects in the subfolder
    objects = client.list_objects(Bucket=bucket_name, Prefix=subfolder_name)

    # Iterate over the objects in the subfolder
    for object in objects["Contents"]:
        # Get the key of the object
        object_key = object["Key"]

        # Download the object to the local directory
        client.download_file(bucket_name, object_key, local_directory + "/" + object_key)

# Print a message to indicate that the download is complete
print("Download complete!")
