from pathlib import Path
from neural_network_model.model import SETTING
import boto3
import paramiko
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# AWS credentials
ACCESS_KEY = SETTING.EC2_SETTING.ACCESS_KEY
SECRET_KEY = SETTING.EC2_SETTING.SECRET_KEY

# EC2 instance details
AMI_ID = SETTING.EC2_SETTING.AMI_ID  # free-tier Ubuntu 18.04 LTS 64-bit
INSTANCE_TYPE = SETTING.EC2_SETTING.INSTANCE_TYPE  # t2.micro
KEY_NAME = SETTING.EC2_SETTING.KEY_NAME  # 'your_key_name'
# SECURITY_GROUP = 'your_security_group'
REGION = SETTING.EC2_SETTING.REGION_NAME  # 'your_region'

# Git repository details
REPO_URL = SETTING.EC2_SETTING.REPO_URL  # 'your_repository_url'
# REPO_FOLDER = 'your_repository_folder'
BRANCH_NAME = SETTING.EC2_SETTING.BRANCH_NAME  # 'your_branch_name'

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

        self.ec2: boto3.Session.client = None
        self.instance_id = None

    def spin_up_instance(self):
        # Connect to AWS
        session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )
        self.ec2 = session.client("ec2")

        # Spin up an EC2 instance
        response = self.ec2.run_instances(
            ImageId=self.ami_id,
            InstanceType=self.instance_type,
            KeyName=self.key_name,
            # SecurityGroups=[SECURITY_GROUP],
            MinCount=1,
            MaxCount=1,
        )
        self.instance_id = response["Instances"][0]["InstanceId"]

    def set_up_security_groups(self):
        # Specify your AWS credentials
        access_key = "YOUR_ACCESS_KEY"
        secret_key = "YOUR_SECRET_KEY"
        region = "us-west-2"  # Specify the region where you want to create the security group

        # Create a Boto3 client for EC2
        ec2_client = boto3.client(
            "ec2",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )

        # Create a security group
        response = ec2_client.create_security_group(
            Description="My Security Group",
            GroupName="my-security-group",
            VpcId="your-vpc-id",
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

    def run_commands(self, *args, **kwargs):
        repo_url = kwargs.get("repo_url")
        repo_folder = kwargs.get("repo_folder")
        branch_name = kwargs.get("branch_name")

        # Wait for the instance to be running
        # ec2.wait_until_running(InstanceIds=[instance_id])
        # print('EC2 instance is running')

        # Get the public IP address of the instance
        response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
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
        self.ec2.terminate_instances(InstanceIds=[self.instance_id])
        print("EC2 instance terminated")


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
    ec2.spin_up_instance()

    # Run commands on the EC2 instance
    # ec2.run_commands(
    #     repo_url=REPO_URL,
    #     repo_folder=SETTING.EC2_SETTING.REPO_FOLDER,
    #     branch_name=BRANCH_NAME
    # )
