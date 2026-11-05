"""Language detection done in plain code, so the reply language never depends on a small model's guess.

en       - English
bn       - Bangla in Bangla script
banglish - Bangla written in English letters
"""
from __future__ import annotations

from .textutil import BN_CHAR, LATIN_CHAR, latin_words

# Romanised-Bangla words that are not English words.
_BANGLISH_STRONG = frozenset(
    """
    ami amar amake amader apni apnar apnake apnader tumi tomar
    koto kotto kibhabe kivabe kibhave kemon kemone kemne kothay kothaye kobe kokhon kokhono kon
    korbo korte korar korchi korini koren korun kori kore korle
    hobe hole hoy hoye hoyeche holo hote hoyna
    lage lagbe laghbe jay jabe jaay pabo pabe pete paoa paowa chai chaichi chacchi
    bhule bhulechi bhulegechi gechi gese gele theke ekta ekti ek bolun bolen janabo janan jante
    nite nibo dite dibo dilam khola khulbo khulte khule thake bochor bosor bochhor ache achhe nai nei
    kinle kine kinbo labh lav bondho bandho bhalo keno kintu ebong tahole
    shomoy somoy boyosh boyos jonno shathe sathe shuru shesh bodlate bodlano bodle bodol poriborton
    tulte tulbo jama rakhte rakhbo hishab hisab tothyo tottho dukkhito
    """.split()
)
# Common in Banglish but short / easy to collide with other text.
_BANGLISH_WEAK = frozenset("er ta gulo ki ke na ar ei eta ota taka din e".split())
_ENGLISH_FUNCTION = frozenset(
    """the is are was were what when how do does did can could will would should i my me you your of in on
    for to a an if there any have has it be with from at by and or that this which who much many""".split()
)


def script_ratio(text: str) -> float:
    """Share of letters that are Bangla script."""
    bn, latin = len(BN_CHAR.findall(text)), len(LATIN_CHAR.findall(text))
    return bn / (bn + latin) if (bn + latin) else 0.0


def banglish_score(text: str) -> float:
    words = latin_words(text)
    strong = sum(1 for w in words if w in _BANGLISH_STRONG)
    weak = sum(1 for w in words if w in _BANGLISH_WEAK)
    return strong + 0.5 * weak


def detect_language(text: str) -> str:
    """Return 'bn', 'banglish' or 'en'."""
    if script_ratio(text) >= 0.3:
        return "bn"
    words = latin_words(text)
    strong = sum(1 for w in words if w in _BANGLISH_STRONG)
    english = sum(1 for w in words if w in _ENGLISH_FUNCTION)
    if strong >= 1 and banglish_score(text) >= english:
        return "banglish"
    return "en"
