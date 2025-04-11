from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Main application settings loaded from environment"""
    google_api_key: str
    db_type: Literal['mysql', 'postgresql'] = 'mysql'
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str
    ai_model: str = 'gemini-2.0-flash-exp'
    max_query_rows: int = 1000

    @property
    def database_url(self) -> str:
        """Construct DB URL for SQLAlchemy / connector usage"""
        return f"{self.db_type}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        case_sensitive = False

def get_settings() -> Settings:
    return Settings()
