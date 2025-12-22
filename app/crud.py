import json
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from . import models
from .auth import get_password_hash


def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, email: str, password: str) -> models.User:
    hashed_password = get_password_hash(password)
    user = models.User(email=email, password_hash=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_analysis(
    db: Session,
    user_id: int,
    text: str,
    label_pred: int,
    prob_ai: float,
    model_name: str,
    segments: List[dict],
) -> models.Analysis:
    analysis = models.Analysis(
        user_id=user_id,
        text=text,
        label_pred=label_pred,
        prob_ai=prob_ai,
        model_name=model_name,
        segments_json=json.dumps(segments),
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_analyses(db: Session, user_id: int, limit: int = 20) -> List[models.Analysis]:
    return (
        db.query(models.Analysis)
        .filter(models.Analysis.user_id == user_id)
        .order_by(models.Analysis.created_at.desc())
        .limit(limit)
        .all()
    )


def get_analysis_by_id(
    db: Session, user_id: int, analysis_id: int
) -> Optional[models.Analysis]:
    return (
        db.query(models.Analysis)
        .filter(
            models.Analysis.user_id == user_id, models.Analysis.id == analysis_id
        )
        .first()
    )


def get_stats(db: Session, user_id: int) -> dict:
    total_count = (
        db.query(func.count(models.Analysis.id))
        .filter(models.Analysis.user_id == user_id)
        .scalar()
    )
    ai_count = (
        db.query(func.count(models.Analysis.id))
        .filter(models.Analysis.user_id == user_id, models.Analysis.label_pred == 1)
        .scalar()
    )
    human_count = (
        db.query(func.count(models.Analysis.id))
        .filter(models.Analysis.user_id == user_id, models.Analysis.label_pred == 0)
        .scalar()
    )
    avg_prob_ai = (
        db.query(func.avg(models.Analysis.prob_ai))
        .filter(models.Analysis.user_id == user_id)
        .scalar()
    )
    return {
        "total_count": total_count or 0,
        "ai_count": ai_count or 0,
        "human_count": human_count or 0,
        "avg_prob_ai": float(avg_prob_ai) if avg_prob_ai is not None else 0.0,
    }
