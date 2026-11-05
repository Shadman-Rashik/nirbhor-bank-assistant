# Loom plan (5 minutes max, scored on clarity for a non-technical listener: 20% of the marks)

Speak to a bank manager, not to an engineer. Say "the assistant", not "the pipeline". Record after you have the real numbers (they are below).
Do a full rehearsal once with a stopwatch; the video is "a core part of the task".

| Time | What to show | What to say (adapt in your own words) |
|---|---|---|
| 0:00-0:40 | README top, or one title slide | The problem: a bank wants a chat assistant, but customer data cannot go to outside AI services, and customers write in English, Bangla and Banglish. Two rules: answer only from the FAQ, and say "I don't know, let me connect you to a person" instead of guessing. |
| 0:40-1:30 | The step diagram in the README | It works like a careful clerk: first understand the language, then look up the FAQ, then write one short reply, then check the numbers against the FAQ. The small model only writes the sentence. Code makes the safety decisions. If the check fails twice it shows the original FAQ text rather than risk a wrong number. |
| 1:30-3:15 | The live demo (see "How to run the demo" below) | Demo in this order: (1) English `What is the late payment fee on a credit card?`; (2) Banglish `Personal loan er interest rate koto?`; (3) Bangla `ক্রেডিট কার্ডের বিল কবে দিতে হয়?`; (4) not in the FAQ, `আজকের স্বর্ণের দাম কত?`, and show it hands over to a person; (5) the tricky one, `Amar boyosh 17, ami ki student loan pabo?`: the FAQ has personal-loan rules for ages 25-60, and a careless assistant would say "you are too young". Ours says it doesn't know. |
| 3:15-4:05 | The results tables (README "Results") | On the 40 given questions: 40/40 correct, 30/30 answerable ones right, 10/10 correct "I don't know", median 0.71 seconds per answer (Colab T4 GPU, warm model). The same 40 on an online model (Claude): also 40/40, so the small model matched it, and the customer data never leaves the bank. Say that the online model was Claude, not independent, and its speed was not measured. |
| 4:05-5:00 | The "what does not work" list in the note | Be plain. (1) 3 of the 40 replies fell back to the English FAQ text, so a Bangla customer got English; facts right, language not. (2) On my own harder 40 questions it scored 37/40, and with the strict check off one near-miss question got an answer when it should have said "I don't know". (3) My answer key was written after reading the test questions, so 40/40 is optimistic; 37/40 is the more honest number. (4) Speed was measured on a Colab GPU, not the GTX 1650. Next: fresh test questions from someone else, a larger model, real hand-over queue. Ending on honest limits scores well. |

## How to run the demo (your PC blocks installs, so do it in Colab)

In the Colab notebook where Ollama is running and the models are pulled, add a new cell and run:

```
import sys; sys.path.insert(0, "src")
from nirbhor.config import load_settings
from nirbhor.factory import build_assistant
bot = build_assistant(load_settings(use_judge=False))
for q in ["What is the late payment fee on a credit card?",
          "Personal loan er interest rate koto?",
          "ক্রেডিট কার্ডের বিল কবে দিতে হয়?",
          "আজকের স্বর্ণের দাম কত?",
          "Amar boyosh 17, ami ki student loan pabo?"]:
    a = bot.answer(q)
    print(f"Q: {q}\n   [{a.lang} | {a.status} | {a.reason} | {a.timings_ms.get('total', '')} ms]\nA: {a.text}\n")
```

Record the screen with that cell's output on screen. The first call can be slow while the model loads: run the cell once before recording.
If the Colab runtime was reset since the last run: re-run the install/start cells and `ollama pull gemma3:4b` and `ollama pull bge-m3`, upload the project zip again.
The browser chat window (`python scripts/serve_ui.py`) needs Ollama on the same computer; show it only if you have that working, and say it runs the same code.

Tips: have everything running before you press record; keep one browser tab; do not read code aloud;
if a live answer is slow, say so and show the measured median instead.

Questions the hiring team may ask on the call (prepare one sentence each): why a fixed pipeline instead of one free agent;
how you chose the refusal threshold; what happens when the number check fails; why the glossary is not "cheating"
(it holds vocabulary, not answers); why the strict check is off; what you would change first with more time.
