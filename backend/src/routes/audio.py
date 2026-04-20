"""Audio file upload routes."""

import os
import uuid
import aiofiles
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from src.models import User, MediaFile
from src.middleware.deps import require_patient
from config.settings import get_settings

router = APIRouter(prefix="/files", tags=["files"])
settings = get_settings()

ALLOWED_EXTENSIONS = settings.allowed_extensions_set
MAX_SIZE_BYTES = settings.AUDIO_MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/audio/upload", status_code=201)
async def upload_audio(
    file: UploadFile = File(...),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    """Direct multipart audio upload. Returns a fileId for use in assessment answers."""
    # Validate extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )
    else:
        ext = ".webm"

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.AUDIO_MAX_FILE_SIZE_MB} MB",
        )

    # Save to local storage
    file_id = str(uuid.uuid4())
    storage_dir = Path(settings.STORAGE_LOCAL_PATH)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_key = f"{file_id}{ext}"
    file_path = storage_dir / storage_key

    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    # Create DB record
    media = MediaFile(
        id=file_id,
        owner_user_id=user.id,
        original_filename=file.filename,
        storage_key=storage_key,
        mime_type=file.content_type,
        file_size=len(content),
        status="available",
    )
    db.add(media)
    await db.flush()

    return {
        "fileId": media.id,
        "status": "available",
        "fileName": file.filename,
        "size": len(content),
    }
