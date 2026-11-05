"""Regression for a glossary bug found through failure analysis.

'পরিবর্তন' (change) used to map only to 'registered mobile number', so a Bangla question about changing the NOMINEE
retrieved the phone-number FAQ. Generic change words and 'mobile number' are now separate concepts.
The questions below are new wordings, not taken from the 40 given questions.
"""
from nirbhor.glossary import expand
from nirbhor.lang import detect_language
from nirbhor.retriever import FaqRetriever


def _top(faq, question):
    lang = detect_language(question)
    return FaqRetriever(faq, None).search(question, expand(question) if lang != "en" else "", k=1)[0].faq_id


def test_change_word_alone_does_not_mean_mobile_number():
    assert "mobile" not in expand("নমিনি পরিবর্তন করতে কী লাগবে?")
    assert "change" in expand("Nominee bodlate ki lagbe?")


def test_nominee_change_goes_to_the_nominee_faq(faq):
    assert _top(faq, "নমিনি পরিবর্তন করতে কী লাগবে?") == 15
    assert _top(faq, "Nominee bodlate ki lagbe?") == 15


def test_mobile_number_change_still_goes_to_the_phone_faq(faq):
    assert _top(faq, "আমার মোবাইল নম্বর বদলাতে চাই, কীভাবে করব?") == 12
    assert _top(faq, "Phone number bodlate ki lagbe?") == 12
