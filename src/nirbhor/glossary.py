"""Banking vocabulary for Bangla and Banglish -> English search terms.

The FAQ is English only, so a Bangla or Banglish question has to be mapped onto English words before keyword
retrieval can work. This is a plain lookup table of general banking words (loan, card, branch, interest ...).
It contains no answers and no facts, only vocabulary. The matched English terms are appended to the question
as extra search text; the original question is never replaced.
"""
from __future__ import annotations

import unicodedata

from .textutil import latin_words

# (English search terms, Bangla-script fragments, romanised-Bangla words)
# Bangla fragments match as substrings (Bangla attaches suffixes: কার্ডের, এটিএমে).
# Romanised words match a token exactly, or as a prefix when the word is 5+ letters (korbo -> korbo, korbona).
CONCEPTS: list[tuple[str, list[str], list[str]]] = [
    ("savings account", ["সেভিংস", "সঞ্চয়ী"], []),
    ("account", ["অ্যাকাউন্ট", "একাউন্ট", "এ্যাকাউন্ট", "হিসাব"], ["hishab", "hisab"]),
    ("open account documents needed", ["খুলতে", "খুলব", "খুলবো", "খোলার", "খুললে"], ["khulbo", "khulte", "khulle"]),
    ("documents required", ["কাগজ", "ডকুমেন্ট", "প্রয়োজনীয়"], ["kagoj", "kagojpotro", "dolil"]),
    ("minimum balance", ["সর্বনিম্ন", "ন্যূনতম", "মিনিমাম"], ["sorbonimno", "shorbonimno", "nunotom"]),
    ("balance", ["ব্যালেন্স"], []),
    ("debit card", ["ডেবিট"], []),
    ("card", ["কার্ড"], []),
    ("credit card", ["ক্রেডিট"], []),
    ("annual fee yearly charge", ["বার্ষিক", "বাৎসরিক", "বছরের ফি"], ["barshik", "batsorik", "bochorik"]),
    ("lost stolen card block hotline", ["হারিয়ে", "হারানো", "হারিয়েছি", "চুরি", "ব্লক"], ["hariye", "haraye", "hariyeche", "haraiche", "churi"]),
    ("PIN reset forgot", ["পিন", "রিসেট", "ভুলে"], ["bhule", "bhulechi", "bhulegechi"]),
    ("NB Go mobile app register", ["অ্যাপ", "রেজিস্টার", "নিবন্ধন", "এনবি গো"], ["registar"]),
    ("daily fund transfer limit per day", ["দিনে", "প্রতিদিন", "লেনদেনের সীমা"], ["protidin", "ek din", "dine"]),
    ("maximum limit", ["সর্বোচ্চ", "সীমা", "লিমিট"], ["sorbocco", "sorboccho", "shorbocco", "shima", "sima"]),
    ("transfer", ["ট্রান্সফার", "পাঠাতে", "পাঠানো"], []),
    ("BEFTN other bank", ["বিইএফটিএন", "অন্য ব্যাংক"], ["onno bank", "onyo bank"]),
    ("ATM withdrawal", ["এটিএম", "তোলা", "তুলতে", "উত্তোলন"], ["tulte", "tulbo", "tola"]),
    ("branch opening hours days", ["শাখা", "শাখাগুলো", "খোলা থাকে", "সময়"], ["shakha", "shomoy", "somoy", "khola"]),
    ("close account closure", ["অ্যাকাউন্ট বন্ধ", "একাউন্ট বন্ধ", "বন্ধ করতে", "বন্ধ করব", "বন্ধ করবো"], ["bondho kor", "bandho kor", "bondho korte"]),
    ("bank statement", ["স্টেটমেন্ট"], []),
    # "change" and "mobile number" are separate ideas: a generic change word must not point at the phone-number FAQ
    ("registered mobile number phone", ["মোবাইল", "ফোন নম্বর", "নম্বর", "নাম্বার"], []),
    ("change update", ["বদল", "বদলাতে", "পরিবর্তন", "আপডেট"], ["bodlate", "bodlano", "bodol", "bodle", "poriborton", "porivartan", "update"]),
    ("dormant inactive account reactivate", ["ডরম্যান্ট", "নিষ্ক্রিয়", "ইনঅ্যাক্টিভ"], []),
    ("joint account holders", ["যৌথ", "জয়েন্ট"], []),
    ("nominee", ["নমিনি", "নমিনী"], []),
    ("FDR fixed deposit", ["এফডিআর", "ফিক্সড", "স্থায়ী আমানত"], []),
    ("early encashment break before maturity", ["ভাঙলে", "ভাঙ্গলে", "ভাঙতে", "মেয়াদের আগে", "আগে তুলে"], ["bhangle", "bhanga", "bhange", "bhangbo", "bhangte"]),
    ("personal loan", ["পার্সোনাল", "ব্যক্তিগত"], []),
    ("loan", ["লোন", "ঋণ", "ঋন"], []),
    ("interest rate", ["সুদ", "সুদের", "ইন্টারেস্ট"], ["sud", "sudh", "sude", "sudher"]),
    ("eligible eligibility age income requirements", ["যোগ্যতা", "যোগ্য", "শর্ত"], ["jogyota", "jogyo", "shorto", "sorto"]),
    ("home loan property value repayment", ["হোম লোন", "বাড়ি", "বাড়ির", "সম্পত্তি", "প্রপার্টি"], ["bari", "barir"]),
    ("income monthly", ["আয়", "ইনকাম"], []),
    ("when date due", ["কবে", "কখন", "তারিখ"], ["kobe", "kokhon", "kokhono", "tarikh"]),
    ("late payment fee penalty missed due date", ["বিলম্ব", "দেরি", "দেরীতে", "জরিমানা"], ["deri", "derite", "joriman", "joriman"]),
    ("transaction I did not make report dispute", ["আমি করিনি", "করিনি", "অননুমোদিত", "অচেনা", "ট্রানজেকশন", "লেনদেন"], ["korini", "kori ni", "anonumodito"]),
    ("where to report complaint hotline", ["অভিযোগ", "জানাব", "জানাবো"], ["janabo", "janate", "janaite", "obhijog", "ovijog"]),
    ("SMS alert charge fee cost", ["এসএমএস", "চার্জ", "খরচ"], ["khoroch", "khorcho"]),
    ("fee", ["ফি"], []),
    ("remittance receive money from abroad", ["রেমিট্যান্স", "বিদেশ", "প্রবাসী"], ["bidesh", "probash", "probashi"]),
    ("cheque book", ["চেক বই", "চেকবই", "চেক"], ["chek"]),
    ("how many days ready time taken", ["কত দিন", "কতদিন", "দিন লাগে"], ["koto din", "kotodin", "din lage"]),
    ("how much amount", ["কত টাকা", "কত"], ["koto"]),
    ("KYC update", ["কেওয়াইসি"], []),
    ("how often every years", ["বছর", "পর পর"], ["bochor", "bosor", "bochhor", "bochhor"]),
    ("age years old", ["বয়স"], ["boyosh", "boyos"]),
    ("hotline call phone", ["হটলাইন", "কল", "ফোন"], []),
    ("NID national id", ["এনআইডি", "জাতীয় পরিচয়"], []),
    ("photo passport", ["ছবি", "পাসপোর্ট"], ["chobi"]),
    ("what should I do steps", ["কী করব", "কি করব", "কী করতে", "কি করতে", "করণীয়"], ["ki korbo", "ki korte", "ki kori"]),
    ("how do I steps procedure", ["কীভাবে", "কিভাবে", "কি ভাবে", "কেমনে"], ["kibhabe", "kivabe", "kibhave", "kemne", "kemone", "kemon kore"]),
    ("what is required need", ["কী লাগবে", "কি লাগবে", "লাগে", "লাগবে"], ["lagbe", "lagve", "lage", "laghbe"]),
]


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def matched_concepts(query: str) -> list[str]:
    """English search terms for every glossary concept found in the query (Bangla or Banglish)."""
    q = _nfc(query).lower()
    words = latin_words(q)
    spaced = " ".join(words)
    found: list[str] = []
    for english, bangla, banglish in CONCEPTS:
        hit = any(_nfc(b) in q for b in bangla)
        if not hit:
            for w in banglish:
                if " " in w:
                    hit = f" {w} " in f" {spaced} "
                else:
                    hit = any(t == w or (len(w) >= 5 and t.startswith(w)) for t in words)
                if hit:
                    break
        if hit:
            found.append(english)
    return found


def expand(query: str) -> str:
    """Space-joined English terms to append to the query for retrieval (empty if nothing matched)."""
    return " ".join(matched_concepts(query))
