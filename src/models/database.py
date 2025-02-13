from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    Float,
)
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    date_occurred = Column(DateTime, default=datetime.utcnow)
    risk_level = Column(Float)
    status = Column(String(50))
    category_id = Column(Integer, ForeignKey("categories.id"))

    category = relationship("Category", back_populates="incidents")


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)

    incidents = relationship("Incident", back_populates="category")


def init_db(db_url: str):
    """Initialize the database with the defined schema"""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine
