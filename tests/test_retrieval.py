from nirbhor.glossary import expand
from nirbhor.lang import detect_language
from nirbhor.retriever import FaqRetriever


def test_every_faq_question_finds_itself(faq):
    r = FaqRetriever(faq, None)
    assert all(r.search(e.question, k=1)[0].faq_id == e.id for e in faq)


def test_glossary_expands_bangla_and_banglish():
    assert "how much" in expand("Personal loan er interest rate koto?")          # koto -> how much
    assert "credit card" in expand("ক্রেডিট কার্ডের বিল কবে দিতে হয়?")            # Bangla script
    assert "ATM withdrawal" in expand("এটিএম থেকে দিনে সর্বোচ্চ কত টাকা তোলা যায়?")
    assert expand("What is the late payment fee on a credit card?") == ""          # plain English: nothing to add


def test_keyword_retrieval_baseline_is_reasonable(faq, test_cases):
    """Keyword-only floor (no embedding model). The real system adds embeddings on top of this."""
    r = FaqRetriever(faq, None)
    ins = [c for c in test_cases if c.expected_faq_id is not None]
    top3 = 0
    for c in ins:
        lang = detect_language(c.question)
        hits = r.search(c.question, expand(c.question) if lang != "en" else "", k=3)
        top3 += c.expected_faq_id in [h.faq_id for h in hits]
    assert top3 / len(ins) >= 0.7
