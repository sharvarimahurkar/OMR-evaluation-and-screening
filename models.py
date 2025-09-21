# models.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class OMRResult(Base):
    __tablename__ = "omr_results"
    id = Column(Integer, primary_key=True, index=True)
    candidate_name = Column(String, nullable=True)
    total_score = Column(Integer, nullable=False)
    per_subject = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ----- DATABASE URL -----
# Use SQLite by default (local file)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./omr_results.db")

# If using SQLite, set check_same_thread
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# create tables (run on import/start)
Base.metadata.create_all(bind=engine)
