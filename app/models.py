from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="USER", nullable=False)
    created_at = Column(DateTime, server_default=func.now(), default=datetime.utcnow)

    analyses = relationship("Analysis", back_populates="user")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(Text, nullable=False)
    label_pred = Column(Integer, nullable=False)
    prob_ai = Column(Float, nullable=False)
    model_name = Column(String, nullable=False)
    segments_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), default=datetime.utcnow)

    user = relationship("User", back_populates="analyses")
