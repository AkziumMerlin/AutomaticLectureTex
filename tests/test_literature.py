from automatic_lecture_tex.literature import LiteratureChunk, retrieve


def test_lexical_retrieval_prefers_matching_math_context():
    chunks = [
        LiteratureChunk(id="a", source="a", text="Банахово пространство и ограниченный оператор"),
        LiteratureChunk(id="b", source="b", text="Интеграл Лебега и измеримые функции"),
    ]
    result = retrieve("ограниченный оператор в банаховом пространстве", chunks, top_k=1)
    assert result[0].id == "a"
