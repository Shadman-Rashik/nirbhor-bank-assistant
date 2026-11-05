"""Prompts. They are in English because small models follow English instructions best."""
from __future__ import annotations

from .faq import FaqEntry

NO_ANSWER = "NO_ANSWER"

_LANG_RULES = {
    "en": "Reply in English.",
    "bn": (
        "Reply in Bangla, written in Bangla script (বাংলা). Keep product names and abbreviations such as "
        "NB Go, NID, BEFTN, RTGS, FDR, OTP, ATM, PIN, KYC, SMS in English letters."
    ),
    "banglish": (
        "Reply in Banglish: Bangla written with English letters only, the way Bangladeshi customers type in chat "
        "(for example 'Apnar account e ...'). Do not use Bangla script. Keep product names and abbreviations "
        "such as NB Go, NID, BEFTN, FDR, OTP, ATM in English."
    ),
}

# Format demonstrations only. The topic (passbooks) is deliberately NOT in the real FAQ.
_EXAMPLES = {
    "en": "",
    "bn": (
        "\nExample of the style (format only, unrelated topic):\n"
        "FAQ: Q: What is the passbook fee? A: The passbook fee is BDT 2,000 per year.\n"
        "Customer: পাসবইয়ের ফি কত?\n"
        "Reply: পাসবইয়ের ফি বছরে BDT 2,000।\n"
    ),
    "banglish": (
        "\nExample of the style (format only, unrelated topic):\n"
        "FAQ: Q: What is the passbook fee? A: The passbook fee is BDT 2,000 per year.\n"
        "Customer: Passbook er fee koto?\n"
        "Reply: Passbook er fee bochore BDT 2,000.\n"
    ),
}


def system_prompt(lang: str, strict: bool = False, problem: str = "") -> str:
    p = (
        "You are the customer-support assistant of Nirbhor Bank PLC.\n"
        "Answer ONLY from the FAQ entry given below. Rules:\n"
        f"1. {_LANG_RULES[lang]}\n"
        "2. Use only facts that are written in the FAQ entry. Never add a fee, number, step or condition "
        "that is not there.\n"
        "3. Copy every number, amount, percentage, phone number, time and name exactly as written "
        "(for example BDT 500, 16999, 3:00 PM, 10 lakh, 7.50%).\n"
        "4. Answer only what the customer asked, in 1 to 3 short sentences.\n"
        f"5. If the FAQ entry does not answer the customer's question, reply with exactly: {NO_ANSWER}\n"
    )
    if strict:
        p += (
            "\nIMPORTANT: your previous answer was rejected"
            + (f" because {problem}" if problem else "")
            + ". Write it again. Use ONLY numbers that appear in the FAQ entry and follow the language rule exactly.\n"
        )
    return p + _EXAMPLES[lang]


def user_prompt(entry: FaqEntry, question: str) -> str:
    return f"FAQ entry:\nQ: {entry.question}\nA: {entry.answer}\n\nCustomer question: {question}\n\nReply:"


def judge_messages(entry: FaqEntry, question: str) -> list[dict]:
    """Strict yes/no check: does this FAQ entry directly answer the customer's question?"""
    system = (
        "You check whether a bank FAQ entry answers a customer's question. The customer may write in English, "
        "Bangla or Banglish (Bangla in English letters). Be strict. Answer NO when the customer asks about a "
        "different product or service than the FAQ entry, about another person's account or private data, about "
        "live market data (gold price, exchange rate, share price), about a place or person the FAQ does not "
        "mention, or when the customer's situation is not covered by the FAQ entry. Reply with one word: YES or NO."
    )
    user = (
        f"FAQ question: {entry.question}\nFAQ answer: {entry.answer}\n\n"
        f"Customer question: {question}\n\nDoes the FAQ entry directly answer the customer's question? YES or NO:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def translate_messages(question: str) -> list[dict]:
    system = "Translate the customer's banking question into plain English. Output only the translation."
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def online_system_prompt(faq: list[FaqEntry]) -> str:
    """Same task for an online model, but it gets the whole FAQ (no retrieval step)."""
    from .templates import IDK

    entries = "\n".join(f"[{e.id}] Q: {e.question}\n    A: {e.answer}" for e in faq)
    return (
        "You are the customer-support assistant of Nirbhor Bank PLC. Customers write in English, Bangla "
        "(Bangla script) or Banglish (Bangla in English letters).\n"
        "Rules:\n"
        "1. Answer ONLY from the FAQ below. Never use outside knowledge.\n"
        "2. Reply in the same language style the customer used: English, Bangla script, or Banglish.\n"
        "3. Copy numbers, amounts and names exactly. Keep the answer to 1 to 3 short sentences.\n"
        "4. If the FAQ does not answer the question, reply with exactly this sentence in the customer's language:\n"
        f"   English: {IDK['en']}\n   Bangla: {IDK['bn']}\n   Banglish: {IDK['banglish']}\n\n"
        f"FAQ:\n{entries}"
    )


# Optional extra for the judge (Settings.judge_examples). Worked examples on topics/wordings that are in neither the
# 40 given questions nor the held-out set, so tuning with them does not contaminate either evaluation.
JUDGE_EXAMPLES = """

Worked examples (for format only):
1. FAQ: What is the annual fee for the debit card? (first year free, BDT 500 from the second year)
   Customer: Credit card er annual fee koto?  -> NO (different product: the FAQ is about the debit card)
2. FAQ: What are the branch opening hours? (Sunday to Thursday, 10:00 AM to 4:00 PM)
   Customer: বৃহস্পতিবার বিকেল ৩টায় শাখায় গেলে কি খোলা পাব?  -> YES
3. FAQ: What is the late payment fee on a credit card? (BDT 500)
   Customer: What is the late fee if I pay my loan instalment after the due date?  -> NO (a loan instalment is not a credit card bill)
4. FAQ: How do I get a cheque book? (at a branch or in the app, BDT 250 for 25 leaves)
   Customer: Can I request a cheque book from the app?  -> YES
5. FAQ: What is the ATM withdrawal limit? (BDT 50,000 per day)
   Customer: Is your ATM open 24 hours?  -> NO (the FAQ does not say)
"""


def build_judge_messages(entry: FaqEntry, question: str, examples: bool = False) -> list[dict]:
    msgs = judge_messages(entry, question)
    if examples:
        msgs[0]["content"] += JUDGE_EXAMPLES
    return msgs
