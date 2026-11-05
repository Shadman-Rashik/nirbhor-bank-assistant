import pytest

from nirbhor.faq import load_cases, parse_key_facts
from nirbhor.lang import detect_language
from nirbhor.metrics import facts_ok, is_idk_text
from nirbhor.verify import check_numbers, language_ok


def test_language_detection_on_given_and_heldout_questions(test_cases, heldout_cases):
    wrong = [(c.id, c.lang, detect_language(c.question)) for c in test_cases + heldout_cases
             if detect_language(c.question) != c.lang]
    assert wrong == []


@pytest.mark.parametrize("text,expected", [
    ("What is today's dollar to taka exchange rate?", "en"),   # 'taka' alone must not make English Banglish
    ("Ek din e koto taka transfer kora jay?", "banglish"),
    ("এটিএম থেকে দিনে সর্বোচ্চ কত টাকা তোলা যায়?", "bn"),
    ("How do I reset my NB Go passcode?", "en"),
])
def test_language_edge_cases(text, expected):
    assert detect_language(text) == expected


def test_numbers_must_come_from_the_faq():
    src = ["Call 16999. A replacement card costs BDT 300 within 7 working days."]
    assert check_numbers("Call 16999 and pay BDT 300.", src).ok
    bad = check_numbers("Call 16999 and pay BDT 350.", src)
    assert not bad.ok and bad.invented == ["350"]


def test_bangla_digits_and_commas_are_normalised():
    src = ["The minimum is BDT 25,000."]
    assert check_numbers("সর্বনিম্ন ২৫,০০০ টাকা", src).ok
    assert not check_numbers("সর্বনিম্ন ২৫,০০১ টাকা", src).ok


def test_time_formats_match():
    assert check_numbers("Before 3:00 PM", ["credited if sent before 3:00 PM"]).ok


def test_reply_language_check():
    assert language_ok("Receiving remittance is free.", "en")
    assert not language_ok("রেমিট্যান্স ফ্রি।", "en")
    assert language_ok("রেমিট্যান্স গ্রহণ করা ফ্রি, NB Go অ্যাপে।", "bn")
    assert not language_ok("Remittance is free.", "bn")
    assert language_ok("Apnar account e koto taka rakhte hobe tar uttor BDT 1,000.", "banglish")
    assert not language_ok("রেমিট্যান্স ফ্রি।", "banglish")


def test_key_fact_matching():
    kf = parse_key_facts("10 lakh|১০ লাখ;5 lakh")
    assert facts_ok("প্রতিদিন ১০ লাখ টাকা এবং 5 lakh", kf)
    assert not facts_ok("up to 10 lakh only", kf)


def test_idk_detection_in_three_languages():
    assert is_idk_text("I'm sorry, I don't know the answer. Let me connect you to a person.")
    assert is_idk_text("দুঃখিত, এই প্রশ্নের উত্তর আমি জানি না।")
    assert is_idk_text("Dukkhito, eta ami jani na.")
    assert not is_idk_text("Receiving remittance is free.")
    assert not is_idk_text("Personal loan er interest rate bochore 12%.")   # "person" inside "Personal" must not count as a hand-over


def test_gold_files_are_consistent(test_cases, heldout_cases, faq):
    ids = {e.id for e in faq}
    for c in test_cases + heldout_cases:
        assert c.lang in ("en", "bn", "banglish")
        assert c.expected_faq_id is None or c.expected_faq_id in ids
        assert (c.expected_faq_id is None) == (c.key_facts == ())
    assert len(test_cases) == 40 and sum(c.expected_faq_id is None for c in test_cases) == 10
