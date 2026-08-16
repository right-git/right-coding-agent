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
    enable_vision_model: bool = Field(
        default=False,
        description="Register the locator-driven screen tools (screen_locate, screen_click) and preload the "
        "LocateAnything vision model. Off by default so the multi-GB model never loads.",
    )
    enable_voice_model: bool = Field(
        default=False,
        description="Start push-to-talk and load the speech models (whisper ASR, Silero TTS). "
        "Off by default so no ASR model loads at startup.",
    )
    vision_quantization: str = Field(
        default="none",
        description="Vision locator weight quantization: 'none' (fp16/bf16) or 'int8' (halves memory).",
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
    voice_tts_speaker_en: str = Field(
        default="en_0",
        description="Silero v3_en speaker used when a sentence has no Cyrillic (en_0 … en_117).",
    )
    voice_language: str | None = Field(
        default=None,
        description="Fixed speech language code such as 'ru'; None auto-detects per utterance.",
    )
    fish_audio_api_key: str | None = Field(default=None, description="Fish Audio API key (planned cloud TTS).")
    fish_audio_tts_model: str = Field(default="s2.1-pro", description="Fish Audio TTS model name.")
    elevenlabs_api_key: str | None = Field(default=None, description="ElevenLabs API key (planned cloud TTS).")


settings = Settings()
