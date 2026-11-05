"""Demo chat app.  streamlit run app.py   (Ollama must be running, see README)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from nirbhor.config import load_settings  # noqa: E402
from nirbhor.factory import build_assistant  # noqa: E402

st.set_page_config(page_title="Nirbhor Bank assistant", page_icon="🏦")
st.title("Nirbhor Bank assistant")
st.caption("Runs fully on this computer. Ask in English, বাংলা or Banglish.")

defaults = load_settings()
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Answer model (Ollama tag)", defaults.gen_model)
    use_emb = st.checkbox("Use embeddings for search", defaults.use_embeddings)
    use_judge = st.checkbox("Strict 'does the FAQ answer this?' check", defaults.use_judge)
    show_steps = st.checkbox("Show how each answer was decided", True)
    st.markdown("**Try:**")
    for ex in ["Personal loan er interest rate koto?", "ক্রেডিট কার্ডের বিল কবে দিতে হয়?",
               "What is the late payment fee on a credit card?", "আজকের স্বর্ণের দাম কত?",
               "Amar boyosh 17, ami ki student loan pabo?"]:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex


@st.cache_resource(show_spinner="Loading the model ...")
def get_bot(model: str, use_emb: bool, use_judge: bool):
    bot = build_assistant(load_settings(gen_model=model, use_embeddings=use_emb, use_judge=use_judge))
    bot.warmup()
    return bot


try:
    bot = get_bot(model, use_emb, use_judge)
except Exception as e:  # Ollama not running / model not pulled
    st.error(f"Could not start the model: {e}\n\nIs Ollama running, and did you run `ollama pull {model}`?")
    st.stop()

if "chat" not in st.session_state:
    st.session_state["chat"] = []
for m in st.session_state["chat"]:
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if m.get("steps") and show_steps:
            st.caption(m["steps"])

q = st.chat_input("Type your question") or st.session_state.pop("pending", None)
if q:
    st.session_state["chat"].append({"role": "user", "text": q})
    with st.chat_message("user"):
        st.write(q)
    a = bot.answer(q)
    steps = (f"language: {a.lang} · result: {a.status} ({a.reason}) · FAQ used: {a.faq_ids or 'none'} · "
             f"{a.total_ms / 1000:.1f}s · best matches: " + ", ".join(f"#{h['faq_id']} ({h['score']})" for h in a.hits))
    st.session_state["chat"].append({"role": "assistant", "text": a.text, "steps": steps})
    with st.chat_message("assistant"):
        st.write(a.text)
        if a.status == "idk":
            st.info("A person will take over this chat.")
        if show_steps:
            st.caption(steps)
