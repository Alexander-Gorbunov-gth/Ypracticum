from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    local_develop: bool = Field(default=True, validation_alias="LOCAL_DEVELOP")
    local_develop_line_limit: int = Field(
        default=30, validation_alias="LOCAL_DEVELOP_LINE_LIMIT"
    )


cfg = Settings()
