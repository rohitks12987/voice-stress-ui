import pymysql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURATION ---
DB_USER = "root"
DB_PASS = "your_password" # Update this
DB_HOST = "localhost"
DB_NAME = "mental_health_db"

# 1. Connect to MySQL to ensure Database exists
temp_engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}")
with temp_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    print(f"✅ Database '{DB_NAME}' verified/created.")
temp_engine.dispose()

# 2. Setup SQLAlchemy for the specific Database
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 3. AUTOMATED TABLE MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True)
    created_at = Column(DateTime, default=datetime.now)

class WellnessLog(Base):
    __tablename__ = "wellness_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    water_intake = Column(Float)
    stress_score = Column(Integer)
    emotion = Column(String(50))
    recorded_at = Column(DateTime, default=datetime.now)

# AUTOMATICALLY CREATE ALL TABLES IN MYSQL
Base.metadata.create_all(bind=engine)
print("✅ All MySQL tables are synced and ready.")

# --- 4. FASTAPI LOGIC ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class WellnessRequest(BaseModel):
    user_id: int
    water: float
    stress: int
    emotion: str

@app.post("/api/save-wellness")
def save_wellness(data: WellnessRequest):
    db = SessionLocal()
    try:
        new_record = WellnessLog(
            user_id=data.user_id,
            water_intake=data.water,
            stress_score=data.stress,
            emotion=data.emotion
        )
        db.add(new_record)
        db.commit()
        return {"status": "success"}
    finally:
        db.close()