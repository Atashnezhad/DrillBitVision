from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import requests
from neural_network_model.model import SETTING

# Load the credentials from the JSON file
credentials = Credentials.from_authorized_user_file(
    SETTING.GOOGLE_DRIVE_SETTING.CREDENTIALS_PATH,
    ["https://www.googleapis.com/auth/drive"],
)

# Build the Drive API service
drive_service = build("drive", "v3", credentials=credentials)

# Specify the folder ID of the folder you want to download
folder_id = SETTING.GOOGLE_DRIVE_SETTING.FOLDER_ID

# Retrieve the list of files in the folder
results = (
    drive_service.files()
    .list(q=f"'{folder_id}' in parents and trashed=false", fields="files(id, name)")
    .execute()
)
files = results.get("files", [])

# Download each file in the folder
for file in files:
    file_id = file["id"]
    file_name = file["name"]
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    response = requests.get(file_url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"Downloaded: {file_name}")

print("Folder download completed.")
