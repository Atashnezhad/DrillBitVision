import logging
import os
from pathlib import Path

import boto3
import paramiko

from neural_network_model.model import SETTING

# Initialize the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS credentials
ACCESS_KEY = SETTING.EC2_SETTING.ACCESS_KEY
SECRET_KEY = SETTING.EC2_SETTING.SECRET_KEY

# EC2 instance details
AMI_ID = SETTING.EC2_SETTING.AMI_ID  # free-tier Ubuntu 18.04 LTS 64-bit
INSTANCE_TYPE = SETTING.EC2_SETTING.INSTANCE_TYPE  # t2.micro
KEY_NAME = SETTING.EC2_SETTING.KEY_NAME  # 'your_key_name'
# SECURITY_GROUP = 'your_security_group'
REGION = SETTING.EC2_SETTING.REGION_NAME  # 'your_region'

# SSH key details
PEM_FILE = SETTING.EC2_SETTING.PEM_FILE_ADDRESS  # 'your_pem_file.pem'
SSH_USERNAME = SETTING.EC2_SETTING.SSH_USER  # 'your_ssh_username'


class MyEC2:
    """
    A class to manage an EC2 instance
    This is used to spin up an EC2 instance,
    clone a Git repository, and run a script on the instance
    """

    def __init__(self, *args, **kwargs):
        self.access_key = kwargs.get("access_key") or SETTING.EC2_SETTING.ACCESS_KEY
        self.secret_key = kwargs.get("secret_key") or SETTING.EC2_SETTING.SECRET_KEY

        self.ami_id = kwargs.get("ami_id") or SETTING.EC2_SETTING.AMI_ID
        self.instance_type = (
            kwargs.get("instance_type") or SETTING.EC2_SETTING.INSTANCE_TYPE
        )
        self.region = kwargs.get("region") or SETTING.EC2_SETTING.REGION_NAME

        self.key_name = kwargs.get("key_name") or SETTING.EC2_SETTING.KEY_NAME
        self.pem_file = kwargs.get("pem_file") or SETTING.EC2_SETTING.PEM_FILE_ADDRESS
        self.ssh_username = kwargs.get("ssh_username") or SETTING.EC2_SETTING.SSH_USER

        self.ec2_resource = None
        self.instance_id = None
        logger.info("EC2 instance is running")

    def spin_up_instance(self):
        # Connect to AWS
        session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )
        self.ec2_resource = session.resource("ec2")

        # Spin up an EC2 instance
        response = self.ec2_resource.create_instances(
            ImageId=self.ami_id,
            InstanceType=self.instance_type,
            KeyName=self.key_name,
            SecurityGroups=[SETTING.EC2_SETTING.SECURITY_GROUP_NAME],
            MinCount=1,
            MaxCount=1,
        )
        self.instance_id = response[0].id
        logger.info(f"EC2 instance {self.instance_id} is running")

    def set_up_security_groups(self):
        # Create a Boto3 client for EC2
        ec2_client = boto3.client(
            "ec2",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        # Create a security group
        response = ec2_client.create_security_group(
            Description=SETTING.EC2_SETTING.SECURITY_GROUP_DESCRIPTION,
            GroupName=SETTING.EC2_SETTING.SECURITY_GROUP_NAME,
            VpcId=SETTING.EC2_SETTING.VPC_ID,
        )

        # Add inbound rule to allow incoming SSH traffic
        ec2_client.authorize_security_group_ingress(
            GroupId=response["GroupId"],
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 22,
                    "ToPort": 22,
                    "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                }
            ],
        )

        # Add outbound rule to allow all traffic
        ec2_client.authorize_security_group_egress(
            GroupId=response["GroupId"],
            IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
        )
        logger.info("Security group created")

    def run_commands(self, *args, **kwargs):
        repo_url = kwargs.get("repo_url") or SETTING.GITHUB_SETTING.REPO_URL
        repo_folder = kwargs.get("repo_folder") or SETTING.GITHUB_SETTING.FOLDER_NAME
        branch_name = kwargs.get("branch_name") or SETTING.GITHUB_SETTING.BRANCH_NAME

        # Create an EC2 client
        ec2_client = boto3.client(
            "ec2",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

        # Wait for the instance to be running
        waiter = ec2_client.get_waiter("instance_running")
        waiter.wait(InstanceIds=[self.instance_id])
        logger.info("EC2 instance is running")

        # Get the public IP address of the instance
        response = ec2_client.describe_instances(InstanceIds=[self.instance_id])
        public_ip = response["Reservations"][0]["Instances"][0]["PublicIpAddress"]

        # SSH into the instance
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(public_ip, username=SSH_USERNAME, key_filename=PEM_FILE)
        print("SSH connection established")

        # Clone the Git repository
        ssh.exec_command(f"git clone {repo_url} {repo_folder}")
        print("Repository cloned")

        # Change directory to the repository folder
        ssh.exec_command(f"cd {repo_folder}")
        print("Changed directory")

        # Checkout the desired branch
        ssh.exec_command(f"git checkout {branch_name}")
        print("Branch checked out")

        # Run your code
        ssh.exec_command("python your_code.py")
        print("Code executed")

        # Close the SSH connection
        ssh.close()

        # Terminate the EC2 instance
        self.terminate_instance()
        print("EC2 instance terminated")

    def terminate_instance(self):
        # Create an EC2 client
        ec2_client = boto3.client(
            "ec2",
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

        # Terminate the EC2 instance
        ec2_client.terminate_instances(InstanceIds=[self.instance_id])
        logger.info(f"EC2 instance {self.instance_id} has been terminated")


if __name__ == "__main__":
    # Spin up an EC2 instance
    ec2 = MyEC2(
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        ami_id=AMI_ID,
        instance_type=INSTANCE_TYPE,
        region=REGION,
        key_name=KEY_NAME,
        pem_file=PEM_FILE,
        ssh_username=SSH_USERNAME,
    )

    # Set up security groups
    # ec2.set_up_security_groups()

    ec2.spin_up_instance()

    # Run commands on the EC2 instance
    ec2.run_commands(
        repo_url=SETTING.GITHUB_SETTING.REPO_URL,
        repo_folder=SETTING.GITHUB_SETTING.FOLDER_NAME,
        branch_name=SETTING.GITHUB_SETTING.BRANCH_NAME,
    )
