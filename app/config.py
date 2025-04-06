import os
from datetime import timedelta

from dotenv import load_dotenv
from flask import current_app

load_dotenv()

class Config:
    """Base configuration."""

    DEBUG = False
    TESTING = False
    MONGO_URI = os.getenv("MONGO_URI")
    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER")
    BCRYPT_ROUNDS = 12
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_KEY = os.getenv("HF_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
