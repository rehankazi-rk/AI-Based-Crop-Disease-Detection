from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.auth import get_current_user
from backend.database.database import get_db
from backend.database.models import Diagnosis, User

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(Diagnosis)
        .filter(Diagnosis.user_id == current_user.id)
        .order_by(Diagnosis.created_at.desc())
        .all()
    )

    return [
        {
            "id": item.id,
            "crop": item.crop,
            "disease": item.disease,
            "confidence": item.confidence,
            # Old rows may have absolute paths; expose only the opaque stored filename.
            "image_path": (
                str(item.image_path).replace("\\", "/").rsplit("/", 1)[-1]
                if item.image_path
                else None
            ),
            "summary": item.disease_summary,
            "created_at": item.created_at.isoformat(),
        }
        for item in records
    ]
