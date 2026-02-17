# app/models.py

from sqlalchemy import Column, Integer, Text, String, DateTime
from datetime import datetime
from app.db.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text)
    assistant_message = Column(Text)
    emotion = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)