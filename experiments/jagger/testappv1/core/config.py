# config.py
import os

class Settings:
    DATABASE_PATH = os.getenv("DATABASE_PATH", "chroma_storage/chroma.db")
