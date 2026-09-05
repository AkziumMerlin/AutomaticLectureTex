from automatic_lecture_tex.asr import is_hotword_prompt_echo


def test_hotword_prompt_echo_detects_verbatim_vocabulary_list() -> None:
    hotwords = ["нормированное пространство", "банахово пространство", "линейный оператор"]
    text = "Нормированное пространство, банахово пространство, линейный оператор."

    assert is_hotword_prompt_echo(text, hotwords)


def test_hotword_prompt_echo_keeps_real_lecture_sentence() -> None:
    hotwords = ["нормированное пространство", "банахово пространство", "линейный оператор"]
    text = "Полное нормированное пространство называется банаховым пространством."

    assert not is_hotword_prompt_echo(text, hotwords)
