"""Read-only single-user Google Drive connector; credentials never enter the repo."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import Config, SourceConfig

SCOPE = ["https://www.googleapis.com/auth/drive.readonly"]
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDES_MIME = "application/vnd.google-apps.presentation"
FOLDER_MIME = "application/vnd.google-apps.folder"
SUPPORTED_MIMES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
    GOOGLE_DOC_MIME: ".pdf",
    GOOGLE_SHEET_MIME: ".pdf",
    GOOGLE_SLIDES_MIME: ".pdf",
}


@dataclass
class DriveItem:
    id: str
    relative_path: str
    mime_type: str
    fingerprint: str
    extension: str


def _imports():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise RuntimeError("Install the optional dependencies: pip install -e '.[drive]'") from exc
    return Request, Credentials, InstalledAppFlow, build, MediaIoBaseDownload


def get_service(config: Config, source: SourceConfig):
    Request, Credentials, InstalledAppFlow, build, _ = _imports()
    credentials_file = config.resolve_path(source.credentials_file)
    token_file = config.resolve_path(source.token_file)
    if not credentials_file.is_file():
        raise RuntimeError(f"Google OAuth credentials missing: {credentials_file}")
    creds = Credentials.from_authorized_user_file(str(token_file), SCOPE) if token_file.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # OAuth launches a local browser; only suitable for a single-user desktop app.
            creds = InstalledAppFlow.from_client_secrets_file(str(credentials_file), SCOPE).run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
        token_file.chmod(0o600)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_items(service, folder_ids: list[str]) -> list[DriveItem]:
    if not folder_ids:
        raise RuntimeError("Set one or more Google Drive folder_ids in config.yaml")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in folder_ids):
        raise RuntimeError("Invalid Google Drive folder ID")
    found: list[DriveItem] = []
    visited: set[str] = set()

    def walk(folder_id: str, prefix: str):
        if folder_id in visited:
            return
        visited.add(folder_id)
        page_token = None
        while True:
            result = service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                pageSize=1000,
                pageToken=page_token,
                fields="nextPageToken,files(id,name,mimeType,modifiedTime,md5Checksum)",
            ).execute()
            for file in result.get("files", []):
                # Names are displayed as metadata, never written as filesystem paths.
                name = file["name"].replace("/", "_").replace("\\", "_")
                rel = f"{prefix}/{name}" if prefix else name
                mime = file["mimeType"]
                if mime == FOLDER_MIME:
                    walk(file["id"], rel)
                elif mime in SUPPORTED_MIMES:
                    found.append(DriveItem(
                        id=file["id"], relative_path=rel, mime_type=mime,
                        fingerprint="|".join([file["id"], file.get("modifiedTime", ""), file.get("md5Checksum", "")]),
                        extension=SUPPORTED_MIMES[mime],
                    ))
            page_token = result.get("nextPageToken")
            if not page_token:
                break

    for folder_id in folder_ids:
        walk(folder_id, "")
    return found


def download_item(service, item: DriveItem, destination: Path):
    _, _, _, _, MediaIoBaseDownload = _imports()
    if item.mime_type.startswith("application/vnd.google-apps."):
        request = service.files().export_media(fileId=item.id, mimeType="application/pdf")
    else:
        request = service.files().get_media(fileId=item.id)
    with destination.open("wb") as stream:
        downloader = MediaIoBaseDownload(stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
