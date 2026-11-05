"""The decision logic, tested with a scripted stand-in for the model (no Ollama needed)."""
from conftest import ScriptedLLM

from nirbhor.templates import IDK


def test_answers_in_english_from_the_right_entry(make_assistant):
    llm = ScriptedLLM(judge="YES", replies=["Receiving remittance is free and credited within 24 hours."])
    a = make_assistant(llm).answer("Is there a fee to receive remittance from abroad?")
    assert (a.status, a.faq_ids, a.lang) == ("answered", [26], "en")
    assert a.fallback is None and a.retries == 0


def test_below_threshold_never_calls_the_model(make_assistant):
    llm = ScriptedLLM()
    a = make_assistant(llm, retrieval_min=0.9).answer("What is the weather today?")
    assert a.status == "idk" and a.reason == "retrieval_below_threshold" and llm.calls == []
    assert a.text == IDK["en"]


def test_judge_rejection_gives_idk_in_the_customers_language(make_assistant):
    llm = ScriptedLLM(judge="NO")
    a = make_assistant(llm, retrieval_min=0.0).answer("Amar boyosh 17, ami ki student loan pabo?")
    assert a.status == "idk" and a.reason == "judge_rejected" and a.text == IDK["banglish"]


def test_model_abstention_gives_idk(make_assistant):
    llm = ScriptedLLM(judge="YES", replies=["NO_ANSWER"])
    a = make_assistant(llm, retrieval_min=0.0).answer("What is the late payment fee on a credit card?")
    assert a.status == "idk" and a.reason == "llm_abstain"


def test_invented_number_triggers_retry_then_passes(make_assistant):
    llm = ScriptedLLM(replies=["The late fee is BDT 800.", "The late fee is BDT 500."])
    a = make_assistant(llm).answer("What is the late payment fee on a credit card?")
    assert a.status == "answered" and a.retries == 1 and "500" in a.text and a.fallback is None


def test_two_failures_fall_back_to_the_faq_text(make_assistant):
    llm = ScriptedLLM(replies=["The late fee is BDT 800.", "It costs BDT 900."])
    a = make_assistant(llm).answer("What is the late payment fee on a credit card?")
    assert a.status == "answered" and a.fallback == "verbatim_faq"
    assert "BDT 500" in a.text and "800" not in a.text and "900" not in a.text


def test_wrong_language_reply_is_rejected(make_assistant):
    llm = ScriptedLLM(replies=["The late fee is BDT 500.", "Deri korle BDT 500 late fee lagbe."])
    a = make_assistant(llm).answer("Credit card er late fee koto?")
    assert a.lang == "banglish" and a.retries == 1 and a.text.startswith("Deri")


def test_judge_can_be_switched_off(make_assistant):
    llm = ScriptedLLM(replies=["Branches open Sunday to Thursday, 10:00 AM to 4:00 PM."])
    a = make_assistant(llm, use_judge=False).answer("What are the branch opening hours?")
    assert a.status == "answered" and a.judge == [] and len(llm.calls) == 1


def test_retrieval_only_mode_returns_faq_text(make_assistant):
    a = make_assistant(None).answer("How do I close my account?")
    assert a.status == "answered" and a.reason == "retrieval_only" and a.faq_ids == [29]
