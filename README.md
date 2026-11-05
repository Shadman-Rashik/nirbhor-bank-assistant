# Nirbhor Bank offline assistant (English / বাংলা / Banglish)

A chat assistant for a bank that answers **only from the bank FAQ**, replies in the language the customer wrote in
(English, Bangla, or Banglish), runs on a small open model **on the bank's own computer**, and says
*"I don't know, let me connect you to a person"* when the FAQ does not cover the question.

"Nirbhor Bank PLC" and "NB Go" are fictional (from the BIAI Round 2 brief, Task 2). No customer data is used.

## How it works

The assistant is a fixed sequence of steps. Only steps 4 and 5 use the small model, and the model never decides on its own
whether to refuse: code decides, and the "I don't know" text is a fixed template.

```
question
  1. detect language ............ code (Bangla script? Banglish words? else English)
  2. find FAQ candidates ........ embeddings (bge-m3) + keywords; a small banking glossary maps Bangla/Banglish words to English
  3. gate: score too low? ....... yes -> "I don't know" (the model is not even called)
  4. judge (optional, OFF in the final configuration, see below): does this FAQ entry answer THIS question? small model, one word YES/NO; NO -> "I don't know"
  5. write the reply ............ small model, from ONE FAQ entry, in the customer's language
  6. verify in code ............. every number must appear in the FAQ entry; language must match.
                                  Retry once; if it still fails, show the original FAQ text instead of a risky translation
```

**When something goes wrong** the assistant never crashes and never guesses. If the model server or the embedding step fails or times out,
the customer gets the same "I don't know, let me connect you to a person" reply (reason `model_error`, with the error text kept for staff);
empty input is answered the same way and very long input is cut at 500 characters. In evaluation such a failure is counted as **wrong**
(never as a correct "I don't know") and is reported separately, so a broken server cannot make the numbers look better.
`run_eval.py` and `run_all.py` check the server and the model names first and print what to install instead of a stack trace.

Why this shape: the FAQ is English-only, but most test questions are Bangla or Banglish, and small models are weak at
Bangla and tend to invent numbers or answer things they should refuse. So the risky decisions (language, refusing,
numbers) are made by code, and the model only writes a short sentence.

## Setup (Windows, 4 GB GPU is enough)

1. Install **Python 3.10+** and **Ollama** (https://ollama.com/download).
2. Pull the models (check the exact tags with `ollama list` / the Ollama library page; they can change):
   ```
   ollama pull gemma3:4b        # the answering model (3.3 GB; tight on a 4 GB GPU, may be partly on CPU)
   ollama pull bge-m3           # the embedding model
   ollama pull qwen2.5:3b       # optional: a second model to compare
   ```
3. Install the Python packages:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. Check the code: `pytest` (43 tests, no model needed: they use a scripted stand-in model and a fake Ollama server on localhost).

## Run

**Everything in one command** (calibration, model comparison, held-out set, three ablations, failure analysis, results table):
```
python scripts/run_all.py --models gemma3:4b --hardware "GTX 1650 4GB, Windows"
```
On Google Colab, open `colab_run.ipynb` instead. The individual steps are listed below.

```
python scripts/chat_cli.py                       # try it in the terminal
python scripts/serve_ui.py                       # the chat window in your browser, http://127.0.0.1:8000 (nothing extra to install)
python scripts/serve_ui.py --retrieval-only --no-embeddings   # look at the UI with no model installed
streamlit run app.py                             # optional older demo (needs: pip install -r requirements-optional.txt)

python scripts/calibrate.py --write              # pick the retrieval threshold on the 40 held-out questions (not the test ones)
python scripts/run_eval.py --gen-model gemma3:4b --hardware "GTX 1650 4GB"      # final configuration (strict check off by default)
python scripts/run_eval.py --set heldout --gen-model gemma3:4b          # my own 40 extra questions
python scripts/run_eval.py --gen-model gemma3:4b --judge --tag ablation_with_judge            # the strict check is off by default; this turns it ON
python scripts/run_eval.py --gen-model gemma3:4b --no-embeddings --tag ablation_keywords_only
python scripts/run_eval.py --gen-model gemma3:4b --judge --judge-examples --tag ablation_judge_examples   # do worked examples help the judge? measure it, don't assume

python scripts/analyze_failures.py               # sorts every wrong answer into a cause -> failures.md in each results folder
python scripts/make_report.py                    # one table of all runs -> results/REPORT.md (copy the note's numbers from here)
```
Each run writes `results/<tag>/answers.csv` (every answer, with a `manual_grade` column to fill by hand: 1 = right, 0 = wrong),
`metrics.json` and `summary.md`. Model load time is measured separately; latency is per question with a warm model.

Online comparison (same 40 questions, the online model gets the whole FAQ in its prompt):
```
set ONLINE_BASE_URL=https://api.openai.com/v1
set ONLINE_API_KEY=your-key-here
set ONLINE_MODEL=the-model-name
python scripts/compare_online.py run
python scripts/compare_online.py table --local results/test_gemma_no_judge
```
Online result in this repo: `results/online/` (Claude Sonnet 5.5, answered in a chat window, not by API, so no speed). No API key? `python scripts/compare_online.py export`, paste the generated prompt into any online chat, save its answers
into `results/online/answers.csv`, then `python scripts/compare_online.py grade`.

Without any model installed you can still run the keyword-only retrieval baseline:
`python scripts/run_eval.py --retrieval-only --no-embeddings` (look at `retrieval_top1` / `retrieval_top3` in `metrics.json`;
the "correct" column is meaningless in that mode because it returns English FAQ text).

## Results

All numbers come from `results/REPORT.md` (made by `scripts/make_report.py`). Automatic scoring against my own answer key; I read through the answers
but did not grade every row by hand, so treat the numbers as slightly optimistic. Hardware: Colab T4 GPU, model already loaded.

| Run | Questions | Judge | Embeddings | Correct | In-scope right | "I don't know" right | False "I don't know" | False answers | Median s |
|---|---|---|---|---|---|---|---|---|---|
| **test_gemma_no_judge (final)** | 40 given | no | bge-m3 | **40/40** | 30/30 | 10/10 | 0 | 0 | 0.71 |
| heldout_gemma_no_judge (final) | 40 held-out | no | bge-m3 | 37/40 | 26/28 | 11/12 | 0 | 1 | 0.55 |
| test_gemma3_4b | 40 given | yes | bge-m3 | 36/40 | 26/30 | 10/10 | 4 | 0 | 0.95 |
| heldout_gemma3_4b | 40 held-out | yes | bge-m3 | 35/40 | 23/28 | 12/12 | 4 | 0 | 0.70 |
| ablation_judge_examples | 40 given | yes + examples | bge-m3 | 39/40 | 29/30 | 10/10 | 1 | 0 | 1.22 |
| ablation_keywords_only | 40 given | yes | keywords only | 27/40 | 17/30 | 10/10 | 13 | 0 | 0.00 |

What the table shows: embeddings matter most (36/40 vs 27/40 with the judge on). The strict check (judge) cost more than it saved, so the final
configuration has it off; the price is one false answer on a near-miss held-out question. Three replies on the given questions fell back to the
original English FAQ text (correct facts, wrong language for a Bangla/Banglish customer). 

### Local model vs an online model (same 40 given questions)

| | Local: gemma3:4b + bge-m3 (on this computer) | Online: Claude Sonnet 5.5 (whole FAQ in the prompt) |
|---|---|---|
| Correct of 40 | 40/40 | 40/40 |
| In-scope right (of 30) | 30/30 | 30/30 |
| Said "I don't know" when it should (of 10) | 10/10 | 10/10 |
| English / Bangla / Banglish | 14/14, 13/13, 13/13 | 14/14, 13/13, 13/13 |
| Median seconds per question | 0.71 (Colab T4, warm model) | not measured (answered in a chat window) |
| Customer data leaves the bank | No | Yes |

Both score the same on these 40 questions, so on accuracy the small model is not worse here. The online model reads the whole FAQ at once, which is easy for a large model;
the point of the local setup is that the same accuracy is reached without sending customer data out. Limits of this comparison: only 40 questions, the answer key is mine,
the online model was Claude (the tool that helped build this) and was run in a chat window, so it is not independent and its speed was not measured.
The local setup needed code to get there (language check, refusal gate, number check); the online model needed only the prompt in `results/online/prompt_for_chat.md`.

## Layout

```
app.py                     Streamlit chat demo
colab_run.ipynb            the whole run on Google Colab (T4 GPU)
src/nirbhor/webapp.py      small local web server for the chat UI (Python standard library only)
src/nirbhor/web/           the page itself: index.html, style.css, app.js (peach / off-white, system fonts, nothing loaded from the internet)
src/nirbhor/lang.py        language detection (code)
src/nirbhor/glossary.py    Bangla/Banglish -> English banking vocabulary (no answers, only words)
src/nirbhor/retriever.py   embeddings + keyword search
src/nirbhor/prompts.py     all prompts
src/nirbhor/pipeline.py    the step-by-step decision logic
src/nirbhor/verify.py      number and language checks
src/nirbhor/templates.py   fixed "I don't know" replies in 3 languages
src/nirbhor/metrics.py     scoring (shared by local and online runs)
scripts/                   eval, calibration, failure analysis, report, online comparison, terminal chat
data/                      the two given files + gold.csv (my labels for the 40) + heldout.csv (my 40 extra questions)
tests/                     43 tests; a scripted stand-in model and a fake Ollama server replace the real ones
docs/                      one-page note, video script, Bangla walkthrough
```

## How "correct" is scored (and its limits)

Out of scope: correct only if the reply is an "I don't know / connect to a person" reply.
In scope: correct only if it is not a refusal, uses the right FAQ entry, contains the key facts listed in `data/gold.csv`
(numbers or key words, Bangla digits normalised), and is in the customer's language.
This is an automatic check. It can miss a reply that has the right numbers but a wrong or awkward sentence, so
**read `answers.csv` and fill `manual_grade` before quoting final numbers.**

## How the final configuration was chosen

Final configuration: **gemma3:4b + bge-m3 embeddings, strict check (judge) OFF**, retrieval threshold from `config/calibrated.json`
(commit that file: without it the built-in default threshold is used and results differ).

Protocol, so the 40 given questions are not used for tuning: (1) the retrieval threshold was calibrated on my held-out questions;
(2) the candidates (two models, strict check on/off) were compared on the **held-out** set and the best one was picked there;
(3) the 40 given questions were then used only to confirm. Everything is in `results/REPORT.md`.

What the runs showed (automatic scores, Colab T4 GPU, one run per setting at temperature 0; full table in Results above):
- The strict check (judge) cost more than it saved. With gemma3:4b it scored 36/40 on the given questions and 35/40 on held-out with the check on,
  and 40/40 and 37/40 with it off. With it on, 4 questions the FAQ covers were refused.
- Switching it off has a price: on the held-out set one near-miss question (a credit-card limit increase, which the FAQ does not cover)
  got an answer instead of "I don't know" (1 false answer, against 4 false refusals with the check on). For a bank a wrong
  answer can be worse than a refusal, so `--judge` is kept as an option and this trade-off is stated openly.
- Worked examples in the judge prompt moved the given-question score from 36/40 to 39/40 (still below 40/40 with no judge at all); this is one run on 40 questions, so a 3-question gap is weak evidence.
- Embeddings matter most: with the judge on, keyword search alone scored 27/40 against 36/40 with embeddings.
- I also tried qwen2.5:3b early on; its result folders are not part of this repo, so I do not quote numbers for it.
- The retrieval threshold was calibrated on the held-out set, so the held-out numbers are slightly optimistic.

## Known limits (be honest about these in the note and video)

- The glossary and the labels in `gold.csv` were written by me after reading the 40 test questions, and the held-out set was written
  by the same person, so both are optimistic. A friend writing 20 fresh questions would be a fairer test.
- One glossary bug was found because test question 26 (Bangla, changing the nominee) failed in the keyword-only baseline: a generic
  "change" word pointed to the phone-number FAQ. I fixed it in general (change and mobile-number are now separate concepts) and added a
  regression test with new wordings, but this means question 26 is no longer a clean, unseen test.
- The "Connect me to a person" button only writes a line to `logs/handover_requests.jsonl`. A real deployment needs a staff queue behind it.
  The chat UI was checked in Chromium (desktop and phone width) only.
- Small models can still write a fluent but wrong Bangla sentence whose numbers are right. The number check cannot see that.
  It also does not catch numbers written as words (for example "দশ লাখ").
- The strict check (judge) is a small model too and is off in the final configuration; without it, near-miss questions are caught only by the retrieval threshold and by the answering model replying NO_ANSWER, which can miss.
- The automatic scorer is lenient in one place: a reply that fell back to the English FAQ text can still pass the language check for a Banglish question (the fallback line starts with a Banglish prefix). The manual grade is the one to quote.
- Each reply is built from one FAQ entry, so a question that needs two entries gets a partial answer.
- No memory of earlier messages; one question at a time.
- Speed is measured on one machine with the model already loaded.

## How AI was used

Claude (Anthropic) was used throughout, as the brief allows. What it did and what I did:

- **Claude wrote** most of the code, the tests, the evaluation scripts, the answer key drafts (`data/gold.csv`, `data/heldout.csv`), the glossary, the chat UI and the first drafts of this README and the note.
- **I did:** chose Task 2 and the design goals (accuracy first, nothing that could disqualify me, honest reporting); set up and ran everything on Google Colab because my PC blocks installs; ran the evaluation, read the results and decided the final configuration with Claude (strict check off); chose the look of the chat UI and supplied the logo; read the answers myself for natural wording and matching language; and sent Claude the results report so the numbers in this README and the note come from `results/REPORT.md`, not from memory.
- **Found and fixed along the way:** a glossary bug that sent a nominee question to the phone-number FAQ (found because test question 26 failed); a scorer bug where the word "person" inside "Personal loan" was counted as a hand-over to a person (found while grading the online model); the first strict-check setting refusing questions the FAQ covers, which is why it is now off.
- **The online model in the comparison is Claude as well** (answered in a chat window with the whole FAQ in the prompt). It is therefore not independent of the tool that helped build this, and its speed was not measured.
