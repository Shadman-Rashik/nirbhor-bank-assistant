# One-page note: Nirbhor Bank offline assistant

**What I built.** A chat assistant that answers customers of a (fictional) bank in English, Bangla or Banglish using only the
30-entry FAQ, runs on a small open model (gemma3:4b through Ollama) with no customer data sent out, and replies
"I don't know, let me connect you to a person" when the FAQ does not cover the question. It is a fixed pipeline: detect language (code)
-> find the FAQ entry (bge-m3 embeddings + keywords) -> refuse if the match is weak -> write the reply from one FAQ entry ->
verify numbers and language in code (retry once, else show the original FAQ text).

**Why it is built this way.** The FAQ is English, but 26 of the 40 test questions are Bangla or Banglish, and small models are weak at
Bangla, invent numbers and rarely refuse. So code makes the risky decisions (language, refusing, numbers) and the model only writes one short sentence.

**How I tested.** The 40 given questions, labelled by me (`data/gold.csv`), plus 40 questions I wrote myself (`data/heldout.csv`,
including near-miss traps) which I used to set the refusal threshold and to choose the configuration; the 40 given questions were then used to confirm.
Scoring is automatic against my answer key. I also read through the answers myself (natural wording, reply language matches the question language)
but did not grade every row by hand, so the numbers are probably a little optimistic. 43 unit tests cover the decision logic with a stand-in model.

**Results (Colab T4 GPU, model already loaded; all numbers from `results/REPORT.md`).**

| | Local: gemma3:4b + bge-m3 (final) | Online: Claude Sonnet 5.5 |
|---|---|---|
| Correct of 40 | 40/40 | 40/40 |
| Said "I don't know" when it should (of 10) | 10/10 | 10/10 |
| Refused something the FAQ covers (of 30) | 0 | 0 |
| Median / p95 seconds per question | 0.71 / 1.34 | not measured |
| Customer data leaves the bank | No | Yes |

Online model: whole FAQ in the prompt, answered in a chat window, so no speed figure; it is Claude, the same tool that helped build this, so the comparison is not independent. On these 40 questions the small local model matched it.
By language (final configuration): English 14/14, Bangla 13/13, Banglish 13/13.
Held-out set (my 40 extra questions): 37/40; 26/28 answerable questions right, 11/12 out-of-scope refused, 1 false answer.
Strict yes/no check by the model ("judge"): with it on, 36/40 on the given questions (4 false refusals) and 35/40 on held-out, so the final configuration has it off;
with worked examples added to the judge prompt it reached 39/40 on the given questions, still not better than off.
Embeddings vs keyword search (both with the judge on): 36/40 vs 27/40, so embeddings matter most (keyword search alone refused 13 answerable questions).
Speed was measured on a Colab T4, not on the GTX 1650 (4 GB): there the 4B model would be slower and may partly run on CPU, which I have not measured.

**What does not work well.**
- 3 of the 40 replies fell back to the original English FAQ text after the model's own answer failed the number or language check.
  The facts are correct, but the text is English, not the customer's language.
- With the judge off, one near-miss question in the held-out set (a credit-card limit increase, not in the FAQ) got an answer instead of "I don't know".
  A wrong answer can be worse than a refusal in a bank, so `--judge` stays available as an option.
- My glossary and labels were written after reading the test questions, so the 40/40 is optimistic; the held-out 37/40 is the more honest number.
  One glossary bug was found through test question 26, so that question is no longer a clean unseen test.
- The number check cannot catch a fluent but wrong Bangla sentence, or numbers written as words.
- One FAQ entry per reply, no conversation memory, one run per setting (temperature 0).

**What I would do next.** Have someone else write fresh test questions; try a larger quantised model if the bank has more memory;
measure speed on the real GTX 1650; add multi-turn memory and a real hand-over queue for staff; log refusals so staff can see which FAQ entries are missing.
