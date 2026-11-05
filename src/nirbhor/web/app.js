(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("form"), input = $("q"), send = $("send");
  const EXAMPLES = [
    "What is the late payment fee on a credit card?",
    "Personal loan er interest rate koto?",
    "ক্রেডিট কার্ডের বিল কবে দিতে হয়?",
    "আজকের স্বর্ণের দাম কত?",
  ];
  // Analog clock: the bank's logo is the dial, hands show the viewer's local time.
  const CLOCK_SVG = (() => {
    let s = '<circle class="ring" cx="50" cy="50" r="47"/><circle class="ring2" cx="50" cy="50" r="42.5"/>';
    for (let i = 0; i < 12; i++) {
      const major = i % 3 === 0, a = (i * 30 * Math.PI) / 180, r1 = major ? 36.5 : 39, r2 = 41.5;
      s += `<line class="tick" stroke-width="${major ? 2.2 : 1.1}" x1="${50 + r1 * Math.sin(a)}" y1="${50 - r1 * Math.cos(a)}" x2="${50 + r2 * Math.sin(a)}" y2="${50 - r2 * Math.cos(a)}"/>`;
    }
    s += '<image href="/static/logo.png" x="26" y="26" width="48" height="48"/>';
    s += '<g data-h><line class="hand-halo" stroke-width="6.2" x1="50" y1="50" x2="50" y2="29"/><line class="hand" stroke-width="3.4" x1="50" y1="50" x2="50" y2="29"/></g>';
    s += '<g data-m><line class="hand-halo" stroke-width="5" x1="50" y1="50" x2="50" y2="17"/><line class="hand" stroke-width="2.4" x1="50" y1="50" x2="50" y2="17"/></g>';
    s += '<g data-s><line class="sec" stroke-width="1.2" x1="50" y1="58" x2="50" y2="14"/></g>';
    s += '<circle class="hub" cx="50" cy="50" r="3.2"/><circle class="hub2" cx="50" cy="50" r="1.4"/>';
    return s;
  })();
  const clocks = Array.from(document.querySelectorAll("[data-clock]"));
  clocks.forEach((c) => { c.innerHTML = CLOCK_SVG; });
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  function tickClock() {
    const d = new Date(), ms = reduced ? 0 : d.getMilliseconds();
    const s = d.getSeconds() + ms / 1000, m = d.getMinutes() + s / 60, h = (d.getHours() % 12) + m / 60;
    clocks.forEach((c) => {
      c.querySelector("[data-h]").setAttribute("transform", `rotate(${h * 30} 50 50)`);
      c.querySelector("[data-m]").setAttribute("transform", `rotate(${m * 6} 50 50)`);
      c.querySelector("[data-s]").setAttribute("transform", `rotate(${s * 6} 50 50)`);
    });
  }
  if (clocks.length) { tickClock(); setInterval(tickClock, reduced ? 1000 : 100); }

  function showDate() {
    const d = new Date(), mon = $("cal-mon"), day = $("cal-day");
    if (!mon || !day) return;
    mon.textContent = d.toLocaleDateString("en", { month: "short" });
    day.textContent = String(d.getDate());
  }
  showDate(); setInterval(showDate, 60000);

  const LANG = { en: "English", bn: "বাংলা", banglish: "Banglish" };
  const REASON = {
    answered: "answered from the FAQ",
    retrieval_only: "FAQ text (no model)",
    retrieval_below_threshold: "no close FAQ entry",
    judge_rejected: "FAQ entry did not fit the question",
    llm_abstain: "model found no answer",
    model_error: "technical problem",
    empty_input: "empty question",
  };
  // Fixed customer-facing texts for the hand-over (reviewed by a person, not written by the model).
  const HANDOVER = {
    en: { button: "Connect me to a person", sent: "Request sent. A person will follow up with you." },
    bn: { button: "আমাকে একজন প্রতিনিধির সাথে যুক্ত করুন", sent: "অনুরোধ পাঠানো হয়েছে। একজন প্রতিনিধি আপনার সাথে যোগাযোগ করবেন।" },
    banglish: { button: "Amake ekjon manusher shathe connect korun", sent: "Request pathano hoyeche. Ekjon manush apnar shathe jogajog korben." },
  };
  let busy = false;

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;   // textContent only: customer and model text is never parsed as HTML
    return n;
  }
  function toBottom() { chat.scrollTop = chat.scrollHeight; }
  function hideWelcome() { const w = $("welcome"); if (w) w.remove(); document.body.classList.add("chatting"); }

  function addUser(text) {
    hideWelcome();
    const m = el("div", "msg user");
    m.appendChild(el("div", "bubble", text));
    chat.appendChild(m); toBottom();
  }
  function addTyping() {
    const m = el("div", "msg bot");
    const b = el("div", "bubble typing");
    b.setAttribute("role", "status"); b.setAttribute("aria-label", "The assistant is typing");
    b.append(el("i"), el("i"), el("i"));
    m.appendChild(b); chat.appendChild(m); toBottom();
    return m;
  }
  function addBot(data, question) {
    const m = el("div", "msg bot" + (data.reason === "model_error" ? " error" : ""));
    m.appendChild(el("div", "bubble", data.text));

    if (data.status === "idk") {
      const t = HANDOVER[data.lang] || HANDOVER.en;
      const box = el("div", "handover");
      const btn = el("button", "", t.button);
      btn.type = "button";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try {
          const r = await fetch("/api/handover", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, lang: data.lang, reason: data.reason }),
          });
          if (!r.ok) throw new Error(r.status);
          box.replaceChildren(el("span", "sent", t.sent));
        } catch (e) { btn.disabled = false; }
      });
      box.appendChild(btn); m.appendChild(box);
    }

    const meta = el("div", "meta");
    meta.appendChild(el("span", "tag", LANG[data.lang] || data.lang));
    meta.appendChild(el("span", "tag", data.faq_ids && data.faq_ids.length ? "FAQ #" + data.faq_ids.join(", #") : "no FAQ used"));
    meta.appendChild(el("span", "tag", REASON[data.reason] || data.reason));
    meta.appendChild(el("span", "tag", (data.ms / 1000).toFixed(1) + " s"));
    m.appendChild(meta);
    chat.appendChild(m); toBottom();
  }
  function addProblem() {
    const m = el("div", "msg bot error");
    m.appendChild(el("div", "bubble", "Something went wrong on our side. Please try again in a moment."));
    chat.appendChild(m); toBottom();
  }

  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    busy = true; send.disabled = true;
    addUser(question); input.value = "";
    const typing = addTyping();
    try {
      const r = await fetch("/api/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!r.ok) throw new Error(r.status);
      const data = await r.json();
      typing.remove(); addBot(data, question);
    } catch (e) {
      typing.remove(); addProblem();
    } finally {
      busy = false; send.disabled = false; input.focus();
    }
  }

  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  for (const text of EXAMPLES) {
    const c = el("button", "chip", text);
    c.type = "button";
    c.addEventListener("click", () => ask(text));
    $("chips").appendChild(c);
  }

  // "Show how each answer was decided" (remembered if the browser allows it)
  const toggle = $("details-toggle");
  function setDetails(on) {
    document.body.classList.toggle("show-details", on);
    toggle.setAttribute("aria-pressed", String(on));
    toggle.textContent = on ? "Hide how each answer was decided" : "Show how each answer was decided";
    try { localStorage.setItem("details", on ? "1" : "0"); } catch (e) { /* storage may be blocked */ }
  }
  toggle.addEventListener("click", () => setDetails(!document.body.classList.contains("show-details")));
  try { if (localStorage.getItem("details") === "1") setDetails(true); } catch (e) { /* ignore */ }

  // Status line
  fetch("/api/health").then((r) => r.json()).then((h) => {
    const s = $("status");
    s.classList.add(h.ready ? "ready" : "error");
    $("status-text").textContent = h.ready ? "Runs on this computer" : "Assistant unavailable";
    if (h.ready && h.model) $("status-model").textContent = "· " + h.model;
  }).catch(() => {
    $("status").classList.add("error");
    $("status-text").textContent = "Assistant unavailable";
  });
})();
