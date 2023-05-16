import boto3
import paramiko
import os

# AWS credentials
ACCESS_KEY = 'your_access_key'
SECRET_KEY = 'your_secret_key'

# EC2 instance details
AMI_ID = 'your_ami_id'
INSTANCE_TYPE = 't2.micro'
KEY_NAME = 'your_key_name'
SECURITY_GROUP = 'your_security_group'
REGION = 'your_region'

# Git repository details
REPO_URL = 'your_repository_url'
REPO_FOLDER = 'your_repository_folder'
BRANCH_NAME = 'your_branch_name'

# SSH key details
PEM_FILE = 'your_pem_file_path'
SSH_USERNAME = 'your_ssh_username'

# Connect to AWS
session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION
)
ec2 = session.client('ec2')

# Spin up an EC2 instance
response = ec2.run_instances(
    ImageId=AMI_ID,
    InstanceType=INSTANCE_TYPE,
    KeyName=KEY_NAME,
    SecurityGroups=[SECURITY_GROUP],
    MinCount=1,
    MaxCount=1
)
instance_id = response['Instances'][0]['InstanceId']

# Wait for the instance to be running
ec2.wait_until_running(InstanceIds=[instance_id])
print('EC2 instance is running')

# Get the public IP address of the instance
response = ec2.describe_instances(InstanceIds=[instance_id])
public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']

# SSH into the instance
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(public_ip, username=SSH_USERNAME, key_filename=PEM_FILE)
print('SSH connection established')

# Clone the Git repository
ssh.exec_command(f'git clone {REPO_URL} {REPO_FOLDER}')
print('Repository cloned')

# Change directory to the repository folder
ssh.exec_command(f'cd {REPO_FOLDER}')
print('Changed directory')

# Checkout the desired branch
ssh.exec_command(f'git checkout {BRANCH_NAME}')
print('Branch checked out')

# Run your code
ssh.exec_command('python your_code.py')
print('Code executed')

# Close the SSH connection
ssh.close()

# Terminate the EC2 instance
ec2.terminate_instances(InstanceIds=[instance_id])
print('EC2 instance terminated')
