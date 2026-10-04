from types import SimpleNamespace

from automatic_lecture_tex import asr as asr_module
from automatic_lecture_tex.asr import OpenAICompatibleASRBackend, is_hotword_prompt_echo
from automatic_lecture_tex.config import ASRConfig, LLMConfig, RuntimeConfig


def test_hotword_prompt_echo_detects_verbatim_vocabulary_list() -> None:
    hotwords = ["нормированное пространство", "банахово пространство", "линейный оператор"]
    text = "Нормированное пространство, банахово пространство, линейный оператор."

    assert is_hotword_prompt_echo(text, hotwords)


def test_hotword_prompt_echo_keeps_real_lecture_sentence() -> None:
    hotwords = ["нормированное пространство", "банахово пространство", "линейный оператор"]
    text = "Полное нормированное пространство называется банаховым пространством."

    assert not is_hotword_prompt_echo(text, hotwords)



def test_openai_compatible_asr_inherits_endpoint_and_reads_word_timestamps(
    tmp_path, monkeypatch
) -> None:
    captured = {}

    class FakeTranscriptions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(
                text="alpha beta",
                words=[
                    SimpleNamespace(word="alpha", start=0.0, end=0.4),
                    SimpleNamespace(word="beta", start=0.5, end=0.9),
                ],
                segments=[],
            )

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions())
    )

    def fake_make_client(**kwargs):
        captured["client"] = kwargs
        return fake_client

    monkeypatch.setattr(asr_module, "make_openai_client", fake_make_client)

    backend = OpenAICompatibleASRBackend(
        ASRConfig(
            backend="openai_compatible",
            model="openai/whisper-large-v3-turbo",
            language="ru",
            hotwords=["Хан-Банах"],
        ),
        RuntimeConfig(),
        LLMConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="secret",
            compatibility_mode="generic",
        ),
    )

    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"RIFFfake")
    segments = backend._transcribe_chunk(audio, shift=120.0, duration=30.0)

    assert captured["client"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["request"]["model"] == "openai/whisper-large-v3-turbo"
    assert captured["request"]["response_format"] == "verbose_json"
    assert "Хан-Банах" in captured["request"]["prompt"]
    assert segments[0].start == 120.0
    assert segments[0].end == 120.9
    assert segments[0].text == "alpha beta"
