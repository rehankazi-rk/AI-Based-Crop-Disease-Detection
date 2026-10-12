import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.database import get_db
from backend.database.models import Diagnosis, User
from backend.services.image_processor import MAX_IMAGE_BYTES, preprocess_image, validate_image_file
from backend.services.predictor import get_predictor

router = APIRouter(prefix="/predictions", tags=["predictions"])
DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR))).expanduser()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("", status_code=status.HTTP_200_OK)
async def predict_disease(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        # Read at most one byte beyond the limit; never buffer an unbounded upload.
        contents = await file.read(MAX_IMAGE_BYTES + 1)
        if len(contents) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image is too large. Maximum size is 5MB.",
            )

        is_valid, error_message, dimensions = validate_image_file(file.filename or "", contents)
        if not is_valid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)

        image = preprocess_image(contents)
        prediction = get_predictor().predict(image)

        extension = Path(file.filename or "").suffix.lower()
        stored_filename = f"{uuid.uuid4().hex}{extension}"
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOAD_DIR / stored_filename
        try:
            # Exclusive creation prevents overwriting even in the unlikely event of a UUID collision.
            with save_path.open("xb") as handle:
                handle.write(contents)

            disease_summary = (
                f"Detected {prediction['disease']} in {prediction['crop']}. "
                f"Model confidence is {prediction['confidence']:.2%}."
            )
            diagnosis = Diagnosis(
                user_id=current_user.id,
                crop=prediction["crop"],
                disease=prediction["disease"],
                confidence=float(prediction["confidence"]),
                # Store only the opaque filename, never an absolute host filesystem path.
                image_path=stored_filename,
                disease_summary=disease_summary,
            )
            db.add(diagnosis)
            db.commit()
            db.refresh(diagnosis)
        except Exception:
            db.rollback()
            save_path.unlink(missing_ok=True)
            raise

        return {
            "crop": prediction["crop"],
            "disease": prediction["disease"],
            "confidence": round(float(prediction["confidence"]), 4),
            "dimensions": {"width": dimensions[0], "height": dimensions[1]},
            "diagnosis_id": diagnosis.id,
            "image_path": stored_filename,
            "summary": disease_summary,
        }
    finally:
        await file.close()
