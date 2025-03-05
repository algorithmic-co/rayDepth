import os
import shutil

from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def upload_file_to_gdrive(
    credentials_path: str,
    parent_directory_id: str,
    folder_name: str,
    local_file: str
) -> str:
    """
    Uploads 'local_file' to a Google Drive subfolder named 'folder_name',
    which is created if it doesn't exist under 'parent_directory_id'.
    Returns a publicly-accessible URL for the uploaded file.

    :param credentials_path: Path to your service account JSON.
    :param parent_directory_id: ID of the parent folder in Drive.
    :param folder_name: Name of the subfolder to create or reuse under parent_directory_id.
    :param local_file: Path to the local file to be uploaded.
    :return: Publicly-accessible URL (webViewLink or webContentLink).
    """
    # Authenticate using a service account JSON and the Drive scope
    gauth = GoogleAuth()
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        credentials_path,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive = GoogleDrive(gauth)

    # Check if subfolder already exists
    folder_id = None
    folder_list = drive.ListFile({
        "q": f"'{parent_directory_id}' in parents and trashed=false"
    }).GetList()
    for item in folder_list:
        if item["title"] == folder_name and item["mimeType"] == "application/vnd.google-apps.folder":
            folder_id = item["id"]
            break

    # If subfolder doesn't exist, create it
    if folder_id is None:
        folder_meta = {
            "title": folder_name,
            "parents": [{"id": parent_directory_id}],
            "mimeType": "application/vnd.google-apps.folder"
        }
        folder = drive.CreateFile(folder_meta)
        folder.Upload()
        folder_id = folder["id"]

    # Upload the file into subfolder
    file_drive = drive.CreateFile({
        "parents": [{"id": folder_id}],
        "title": os.path.basename(local_file),
    })
    file_drive.SetContentFile(local_file)
    file_drive.Upload()

    # Make file publicly readable
    file_drive.InsertPermission({
        "type": "anyone",
        "value": "anyone",
        "role": "reader"
    })

    # Refresh metadata so we can get the final link
    file_drive.FetchMetadata()
    link = file_drive.get("webContentLink")
    print(f"Uploaded '{local_file}' to folder '{folder_name}' (id={folder_id}).")
    print(f"Public download URL: {link}")
    return link


def zip_dir(input_dir: str, zip_path: str):
    """
    Zip the entire into a file at zip_path (e.g. "/tmp/data.zip").
    """
    base_name, _ = os.path.splitext(zip_path)
    shutil.make_archive(base_name, "zip", input_dir)
