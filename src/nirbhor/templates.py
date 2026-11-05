"""Fixed, hand-written replies. The 'I don't know' message is never written by the model, so it can't drift."""
from __future__ import annotations

IDK = {
    "en": "I'm sorry, I don't know the answer to that from our FAQ. Let me connect you to a person who can help.",
    "bn": "দুঃখিত, এই প্রশ্নের উত্তর আমি জানি না। আমি আপনাকে একজন প্রতিনিধির সাথে যুক্ত করছি।",
    "banglish": "Dukkhito, ei proshner uttor ami jani na. Ami apnake ekjon manusher (representative) shathe connect kore dichhi.",
}

# Used only when the model's translated answer fails the number/language check twice:
# we show the original FAQ text rather than risk a wrong figure.
FALLBACK_PREFIX = {
    "en": "",
    "bn": "আমাদের FAQ অনুযায়ী (ইংরেজিতে): ",
    "banglish": "Amader FAQ onujayi (English e): ",
}

LANG_NAME = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
