from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = Field(..., description="Active runtime environment name.")
    llm_api_key: str = Field(
        ...,
        description="LLM API Key",
    )
    llm_api_base: str = Field(
        ...,
        description="LLM API Base",
    )
    llm_default_model: str = Field(
        default="google/gemini-3.7-flash",
        description="Model the chat starts with; any provider-prefixed id.",
    )
    voice_ptt_key: str = Field(
        default="alt_r",
        description="Push-to-talk toggle key: a pynput key name (alt_r, f8, pause) or a single character.",
    )
    voice_asr_model: str = Field(
        default="large-v3-turbo",
        description="faster-whisper model for the local ASR provider.",
    )
    voice_tts_speaker: str = Field(
        default="xenia",
        description="Silero speaker for the local TTS provider (aidar, baya, kseniya, xenia, eugene).",
    )
    voice_language: str | None = Field(
        default=None,
        description="Fixed speech language code such as 'ru'; None auto-detects per utterance.",
    )
    fish_audio_api_key: str | None = Field(default=None, description="Fish Audio API key (planned cloud TTS).")
    fish_audio_tts_model: str = Field(default="s2.1-pro", description="Fish Audio TTS model name.")
    elevenlabs_api_key: str | None = Field(default=None, description="ElevenLabs API key (planned cloud TTS).")


settings = Settings()
