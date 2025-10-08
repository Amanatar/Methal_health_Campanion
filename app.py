
import os
import re
import time
import datetime as dt
from typing import Dict, List, Optional

import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import pandas as pd
import altair as alt

# Gemini SDK (optional; app works without it using offline fallback)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

load_dotenv()

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title=" Mental Health Companion", page_icon="💙", layout="centered")

# ---------------------------
# Lightweight mood detection
# ---------------------------
@st.cache_resource
def get_vader():
    return SentimentIntensityAnalyzer()

EMOTION_LEXICON = {
    "anxious": {
        "anxious", "anxiety", "worry", "worried", "nervous", "panic", "panicking", "overthink", "overthinking", "uneasy",
        "jitters", "scared"
    },
    "stressed": {
        "stressed", "stress", "overwhelmed", "overwhelm", "pressure", "deadline", "deadlines", "finals", "exams", "tension"
    },
    "sad": {
        "sad", "down", "blue", "depressed", "cry", "crying", "upset", "hopeless", "empty", "numb", "heartbroken"
    },
    "lonely": {
        "lonely", "alone", "isolated", "isolating", "nobody", "no one", "left out"
    },
    "angry": {
        "angry", "mad", "furious", "annoyed", "irritated", "pissed", "rage"
    },
    "frustrated": {
        "frustrated", "stuck", "blocked", "can't focus", "not working"
    },
    "tired": {
        "tired", "exhausted", "drained", "fatigued", "sleepy", "burned out", "burnout", "worn out"
    },
    "guilt": {
        "guilty", "guilt", "ashamed", "shame", "regret", "regretting", "sorry"
    },
    "fear": {
        "afraid", "scared", "fear", "terrified", "frightened", "worried sick"
    },
    "grief": {
        "grief", "grieving", "loss", "passed away", "funeral"
    },
}

CRISIS_PATTERNS = [
    r"\bkill myself\b",
    r"\bkill\s+myself\b",
    r"\bsuicide\b",
    r"\bself[-\s]?harm\b",
    r"\bhurt myself\b",
    r"\bcut myself\b",
    r"\boverdose\b",
    r"\bi (?:don't|do not) want to live\b",
    r"\bi can't go on\b",
    r"\bi(?:'| )?m going to end it\b",
    r"\bi(?:'| )?d be better off dead\b",
    r"\bi want to die\b",
    r"\btake my life\b",
]
CRISIS_REGEX = re.compile("|".join(CRISIS_PATTERNS), flags=re.IGNORECASE)

def detect_crisis(text: str) -> Optional[str]:
    if not text:
        return None
    m = CRISIS_REGEX.search(text)
    return m.group(0) if m else None

def analyze_mood(text: str) -> Dict:
    vader = get_vader()
    vs = vader.polarity_scores(text or "")
    compound = vs.get("compound", 0.0)
    if compound > 0.2:
        sentiment = "positive"
    elif compound < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    text_l = (text or "").lower()
    tokens = re.findall(r"[a-z']+", text_l)
    counts = {emo: 0 for emo in EMOTION_LEXICON}
    for emo, words in EMOTION_LEXICON.items():
        for w in words:
            if " " in w:
                if w in text_l:
                    counts[emo] += 1
            else:
                counts[emo] += sum(1 for t in tokens if t == w)

    top_emotion = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "uncertain"
    top_count = counts.get(top_emotion, 0)
    intensity = abs(compound)
    confidence = min(1.0, 0.25 + intensity + (0.12 * top_count))

    return {
        "sentiment": sentiment,
        "compound": compound,
        "emotion": top_emotion if top_count > 0 else "uncertain",
        "confidence": round(confidence, 2),
        "counts": counts,
    }

# ---------------------------
# Relaxation tips
# ---------------------------
def tip_for_emotion(emotion: str) -> Dict[str, List[str]]:
    if emotion in ("anxious", "fear"):
        return {"title": "Box breathing (4-4-4-4)", "steps": [
            "Breathe in through your nose for 4 seconds.",
            "Hold your breath for 4 seconds.",
            "Exhale through your mouth for 4 seconds.",
            "Hold for 4 seconds, then repeat 3–4 times."
        ]}
    if emotion in ("stressed", "frustrated"):
        return {"title": "2-minute reset", "steps": [
            "Stand up, roll shoulders back and down.",
            "Inhale for 4, exhale for 6 — repeat 5 cycles.",
            "Jot one tiny next step you can do in 5 minutes."
        ]}
    if emotion in ("sad", "grief"):
        return {"title": "Self-compassion pause", "steps": [
            "Place a hand on your chest and notice your breath.",
            "Silently say: “This is hard. I’m not alone in feeling this.”",
            "Ask: “What’s one kind thing I can do for myself today?”"
        ]}
    if emotion == "lonely":
        return {"title": "Gentle reach-out", "steps": [
            "Message one friend or classmate: “Hey, want a short call or a walk?”",
            "If that feels hard, start with a text like: “Thinking of you — how are you?”",
            "Consider a low-key campus event or study group."
        ]}
    if emotion == "angry":
        return {"title": "Tension release quickie", "steps": [
            "Clench fists and shoulders for 5 seconds, then release.",
            "Exhale with a sigh; repeat 3 times.",
            "Write down what you can control about the situation."
        ]}
    if emotion == "tired":
        return {"title": "20-20-20 break", "steps": [
            "Stand up and look 20 feet away for 20 seconds.",
            "Shake out arms and legs; sip water.",
            "Plan a realistic stop time for today."
        ]}
    return {"title": "5-4-3-2-1 grounding", "steps": [
        "Name 5 things you can see.",
        "Name 4 things you can feel.",
        "Name 3 things you can hear.",
        "Name 2 things you can smell.",
        "Name 1 thing you can taste."
    ]}

# ---------------------------
# Crisis response
# ---------------------------
def crisis_support_message() -> str:
    return (
        "I’m really sorry you’re going through this. You’re not alone, and your feelings matter. "
        "If you’re in immediate danger or think you might act on these thoughts, please contact your local emergency number now.\n\n"
        "You can also reach a crisis line for immediate support:\n"
        "- U.S.: call or text 988 (Suicide & Crisis Lifeline)\n"
        "- U.K. & ROI: Samaritans at 116 123\n"
        "- Canada: 1-833-456-4566\n"
        "- Australia: Lifeline at 13 11 14\n"
        "Find more countries at https://findahelpline.com\n\n"
        "If you’d like, tell me your country so I can share more local options. "
        "I’m here with you — would you like to share a bit more about what’s been hardest today?"
    )

# ---------------------------
# LLM: Gemini
# ---------------------------
def build_system_prompt(mood: Dict) -> str:
    return (
        "You are a supportive, non-judgmental mental health companion for students. "
        "Listen with empathy, reflect what you heard, validate feelings, offer one brief, actionable coping idea, "
        "and ask exactly one gentle follow-up question. "
        "Avoid diagnoses or medical instructions. Keep it concise (6–8 sentences). "
        f"Detected mood (for your awareness, no need to state explicitly): "
        f"sentiment={mood.get('sentiment')}, emotion={mood.get('emotion')}."
    )

def call_gemini_response(user_input: str, history: List[Dict], mood: Dict, api_key: str, model: str = "gemini-1.5-flash") -> str:
    if not GEMINI_AVAILABLE or not api_key:
        return offline_response(user_input, mood)

    # Configure once per call (simple + safe)
    genai.configure(api_key=api_key)

    # Keep limited history
    tail = history[-6:] if len(history) > 6 else history[:]

    ghistory = []
    for m in tail:
        if m["role"] == "user":
            ghistory.append({"role": "user", "parts": [m["content"]]})
        elif m["role"] == "assistant":
            ghistory.append({"role": "model", "parts": [m["content"]]})

    sys_prompt = build_system_prompt(mood)
    gmodel = genai.GenerativeModel(model_name=model, system_instruction=sys_prompt)

    try:
        chat = gmodel.start_chat(history=ghistory)
        resp = chat.send_message(
            user_input,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=400,
            )
        )
        text = (resp.text or "").strip()
        return text if text else offline_response(user_input, mood)
    except Exception:
        return f"(Gemini error; falling back)\n\n{offline_response(user_input, mood)}"

def offline_response(user_input: str, mood: Dict) -> str:
    emo = mood.get("emotion", "uncertain")
    sent = mood.get("sentiment", "neutral")
    tip = tip_for_emotion(emo)
    reflection = "Thanks for sharing that. It sounds like you're carrying a lot" if sent == "negative" else \
                 "I appreciate you checking in" if sent == "neutral" else \
                 "I’m glad there are some bright spots"
    emotion_line = f" — and there may be some {emo} in the mix" if emo != "uncertain" else ""
    steps = "\n".join([f"- {s}" for s in tip["steps"]])

    follow_up = "What feels like the next tiny step that’s doable for you?"
    if emo in ("anxious", "stressed"):
        follow_up = "What’s one small thing we could do in the next 5–10 minutes to ease this a bit?"
    elif emo in ("sad", "lonely"):
        follow_up = "Would connecting with someone you trust feel okay, or should we brainstorm a gentle alternative?"

    return (
        f"{reflection}{emotion_line}. I’m here with you.\n\n"
        f"A small thing you could try right now: {tip['title']}\n{steps}\n\n"
        f"{follow_up}"
    )

# ---------------------------
# Streamlit state
# ---------------------------
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "moods" not in st.session_state:
        st.session_state.moods = []
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = True
    if "model" not in st.session_state:
        st.session_state.model = "gemini-1.5-flash"

init_state()

# ---------------------------
# Sidebar: settings and tracker (no API key input box)
# ---------------------------
st.sidebar.subheader("Settings")

# Read key from secrets or env only; do NOT show it in the UI
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

use_llm = st.sidebar.toggle("Use LLM responses (fallback to offline if off or key missing)", value=True)
st.session_state.use_llm = use_llm

gemini_model = st.sidebar.selectbox(
    "Gemini model",
    options=["gemini-1.5-flash", "gemini-1.5-pro"],
    index=0
)
st.session_state.model = gemini_model

# Status pill (no key shown)
status = "Connected ✅" if (GEMINI_API_KEY and GEMINI_AVAILABLE) else "Not configured — using offline responses"
st.sidebar.caption(f"Gemini status: {status}")

st.sidebar.divider()
st.sidebar.subheader("Session controls")
if st.sidebar.button("Start new conversation"):
    st.session_state.messages = []
    st.session_state.moods = []
    st.rerun()

export = st.sidebar.button("Export mood log (CSV)")
if export:
    if st.session_state.moods:
        df = pd.DataFrame(st.session_state.moods)
        fname = f"mood_log_{int(time.time())}.csv"
        df.to_csv(fname, index=False)
        st.sidebar.success(f"Saved {fname}")
    else:
        st.sidebar.info("No mood data to export yet.")

st.sidebar.divider()
st.sidebar.caption("This tool supports wellbeing but isn’t a substitute for professional care. If you’re in immediate danger, call your local emergency number.")

# ---------------------------
# Header
# ---------------------------
st.title("Student Mental Health Companion 💙")
st.caption("A supportive space to check in, reflect, and find small steps forward. If you’re in immediate danger, please contact your local emergency number.")

# ---------------------------
# Render chat history
# ---------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("meta") and m["role"] == "assistant":
            mood_meta = m["meta"]
            st.caption(
                f"Mood reading • Sentiment: {mood_meta.get('sentiment')} (score {mood_meta.get('compound'):.2f}) • "
                f"Emotion: {mood_meta.get('emotion')} (conf {mood_meta.get('confidence')})"
            )
            tip = tip_for_emotion(mood_meta.get("emotion", "uncertain"))
            with st.expander(f"Try this: {tip['title']}"):
                for step in tip["steps"]:
                    st.write(f"- {step}")

# ---------------------------
# Chat input
# ---------------------------
user_input = st.chat_input("How are you feeling today?")
if user_input:
    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Mood + crisis check
    mood = analyze_mood(user_input)
    st.session_state.moods.append({
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "compound": mood["compound"],
        "sentiment": mood["sentiment"],
        "emotion": mood["emotion"],
        "confidence": mood["confidence"],
    })

    crisis_hit = detect_crisis(user_input)
    if crisis_hit:
        reply = crisis_support_message()
        with st.chat_message("assistant"):
            st.write(reply)
            st.warning("If you're in immediate danger, please contact your local emergency number right now.")
            st.link_button("Find a helpline", "https://findahelpline.com")
        st.session_state.messages.append({"role": "assistant", "content": reply, "meta": mood})
        st.stop()

    # Assistant response (Gemini or offline)
    if use_llm and GEMINI_API_KEY and GEMINI_AVAILABLE:
        assistant_text = call_gemini_response(user_input, st.session_state.messages, mood, GEMINI_API_KEY, model=gemini_model)
    else:
        if use_llm and (not GEMINI_API_KEY or not GEMINI_AVAILABLE):
            st.info("Gemini not configured — using offline responses for now.")
        assistant_text = offline_response(user_input, mood)

    with st.chat_message("assistant"):
        st.write(assistant_text)
        st.caption(
            f"Mood reading • Sentiment: {mood.get('sentiment')} (score {mood.get('compound'):.2f}) • "
            f"Emotion: {mood.get('emotion')} (conf {mood.get('confidence')})"
        )
        tip = tip_for_emotion(mood.get("emotion", "uncertain"))
        with st.expander(f"Try this: {tip['title']}"):
            for step in tip["steps"]:
                st.write(f"- {step}")

    st.session_state.messages.append({"role": "assistant", "content": assistant_text, "meta": mood})

# ---------------------------
# Mood tracker chart
# ---------------------------
if st.session_state.moods:
    st.divider()
    st.subheader("Your mood over time")
    df = pd.DataFrame(st.session_state.moods)
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("compound:Q", title="Sentiment score (-1 to 1)"),
            color=alt.value("#3b82f6")
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
