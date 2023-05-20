import os
import pathlib
from typing import List

import boto3

from neural_network_model.model import SETTING


class MyS3:
    """
    This class is used to download the images from the S3 bucket
    """

    # TODO: add the upload method

    def __init__(self, *args, **kwargs):
        self.access_key: str = (
            SETTING.S3_BUCKET_SETTING.AWS_S3_ACCESS_KEY or kwargs.get("ACCESS_KEY")
        )
        self.secret_key: str = (
            SETTING.S3_BUCKET_SETTING.AWS_S3_SECRET_KEY or kwargs.get("SECRET_KEY")
        )
        self.bucket_name: str = SETTING.S3_BUCKET_SETTING.BUCKET_NAME or kwargs.get(
            "BUCKET_NAME"
        )
        self.parent_folder_name: str = (
            SETTING.S3_BUCKET_SETTING.PARENT_FOLDER_NAME or kwargs.get("FOLDER_NAME")
        )
        self.subfolder_name: List[
            str
        ] = SETTING.S3_BUCKET_SETTING.SUBFOLDER_NAME or kwargs.get("SUBFOLDERS")
        self.download_location_address = (
            SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION
            or kwargs.get("DOWNLOAD_LOCATION")
        )
        self.region: str = SETTING.S3_BUCKET_SETTING.REGION_NAME or kwargs.get("region")

    def download_files_from_subfolders(
        self, bucket_name, subfolders, download_location_address
    ):
        # check if the download location exists, if not create it
        if not os.path.exists(self.download_location_address):
            pathlib.Path(self.download_location_address).mkdir(
                parents=True, exist_ok=True
            )

        s3 = boto3.client(
            "s3",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        for subfolder in subfolders:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=subfolder)

            # make a directory for the subfolder
            pathlib.Path(download_location_address / subfolder.split("/")[-1]).mkdir()

            if "Contents" in response:
                for obj in response["Contents"]:
                    file_key = obj["Key"]
                    file_name = file_key.split("/")[-1]
                    subfolder_name = file_key.split("/")[-2]
                    file_path = download_location_address / subfolder_name / file_name
                    s3.download_file(bucket_name, file_key, file_path)


if __name__ == "__main__":
    obj = MyS3()
    # obj.download()

    # Specify your bucket name and subfolders
    bucket_name = SETTING.S3_BUCKET_SETTING.BUCKET_NAME
    subfolders = SETTING.S3_BUCKET_SETTING.SUBFOLDER_NAME
    download_location_address = SETTING.S3_BUCKET_SETTING.DOWNLOAD_LOCATION

    # Download the files from the subfolders
    obj.download_files_from_subfolders(
        bucket_name, subfolders, download_location_address
    )
