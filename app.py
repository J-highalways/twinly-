import streamlit as st
from main import ask, build_database, collection

st.set_page_config(page_title="Twinly", page_icon="🎀", layout="wide")

st.markdown("""
<style>
@keyframes floatUp {
    0% { transform: translateY(0); opacity: 0; }
    15% { opacity: 0.6; }
    100% { transform: translateY(-100vh); opacity: 0; }
}
@keyframes bob {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

.stApp {
    background: linear-gradient(160deg, #ffc9dd 0%, #ffe4ec 45%, #fff5f8 100%);
}

.bubble {
    position: fixed;
    bottom: -40px;
    border-radius: 50%;
    background: rgba(255,255,255,0.5);
    animation: floatUp linear infinite;
    z-index: 0;
}
.b1 { left: 8%;  width: 22px; height: 22px; animation-duration: 14s; }
.b2 { left: 30%; width: 14px; height: 14px; animation-duration: 11s; animation-delay: 3s; }
.b3 { left: 55%; width: 26px; height: 26px; animation-duration: 16s; animation-delay: 1s; }
.b4 { left: 78%; width: 16px; height: 16px; animation-duration: 12s; animation-delay: 5s; }
.b5 { left: 92%; width: 20px; height: 20px; animation-duration: 15s; animation-delay: 2s; }

.header-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    margin-bottom: -10px;
}
.mascot-wrap { animation: bob 2.4s ease-in-out infinite; }

h1 {
    color: #c2185b !important;
    font-weight: 800 !important;
    margin-bottom: 0 !important;
}
.stCaption, p, span, label, div { color: #7a2e4d !important; }

[data-testid="stChatMessage"] {
    background-color: #ffffffee;
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 10px;
    border: 1px solid #ffcfe0;
    box-shadow: 0px 2px 8px rgba(214, 51, 132, 0.12);
    position: relative;
    z-index: 1;
}

.stChatInput textarea, .stChatInput input {
    background-color: #ffffff !important;
    color: #5a2340 !important;
    border-radius: 14px !important;
    border: 2px solid #ffabc9 !important;
    caret-color: #d63384 !important;
}
.stChatInput textarea::placeholder { color: #c97ba0 !important; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffc1d9, #ffe4ec); }
[data-testid="stChatMessageAvatarUser"] { background-color: #ff8fb3 !important; }
[data-testid="stChatMessageAvatarAssistant"] { background-color: #d63384 !important; }

.block-container { position: relative; z-index: 1; }
</style>

<div class="bubble b1"></div>
<div class="bubble b2"></div>
<div class="bubble b3"></div>
<div class="bubble b4"></div>
<div class="bubble b5"></div>
""", unsafe_allow_html=True)

# --- Twinly character (original round blob mascot) ---
twinly_svg = """
<div class="mascot-wrap">
<svg width="100" height="100" viewBox="0 0 200 200">
  <ellipse cx="100" cy="185" rx="50" ry="8" fill="#ffb6d0" opacity="0.35"/>

  <!-- stubby arms -->
  <ellipse cx="35" cy="120" rx="16" ry="10" fill="#ff9ec7" stroke="#ff7fae" stroke-width="2"/>
  <ellipse cx="165" cy="120" rx="16" ry="10" fill="#ff9ec7" stroke="#ff7fae" stroke-width="2"/>

  <!-- round blob body -->
  <circle cx="100" cy="105" r="75" fill="#ffb3d1" stroke="#ff7fae" stroke-width="3"/>

  <!-- little feet -->
  <ellipse cx="75" cy="172" rx="16" ry="9" fill="#ff7fae"/>
  <ellipse cx="125" cy="172" rx="16" ry="9" fill="#ff7fae"/>

  <!-- blush -->
  <circle cx="55" cy="120" r="11" fill="#ff5c98" opacity="0.5"/>
  <circle cx="145" cy="120" r="11" fill="#ff5c98" opacity="0.5"/>

  <!-- big glossy eyes -->
  <ellipse cx="75" cy="100" rx="12" ry="15" fill="#3a1530"/>
  <ellipse cx="125" cy="100" rx="12" ry="15" fill="#3a1530"/>
  <circle cx="79" cy="92" r="4" fill="#ffffff"/>
  <circle cx="129" cy="92" r="4" fill="#ffffff"/>
  <circle cx="72" cy="106" r="2" fill="#ffffff"/>
  <circle cx="122" cy="106" r="2" fill="#ffffff"/>

  <!-- smile -->
  <path d="M85 128 Q100 140 115 128" stroke="#c2185b" stroke-width="3" fill="none" stroke-linecap="round"/>

  <!-- bow on top -->
  <path d="M85 40 Q70 25 55 40 Q70 48 85 40 Z" fill="#ff4f8b" stroke="#e0356f" stroke-width="2"/>
  <path d="M85 40 Q100 25 115 40 Q100 48 85 40 Z" fill="#ff4f8b" stroke="#e0356f" stroke-width="2"/>
  <circle cx="85" cy="40" r="6" fill="#ff2f79"/>
</svg>
</div>
"""

st.markdown(f"""
<div class="header-row">
  {twinly_svg}
  <div>
    <h1>Twinly</h1>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")

if collection.count() == 0:
    with st.spinner("setting things up..."):
        build_database()

# --- session state: chat history + context-engineering state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_summary" not in st.session_state:
    st.session_state.context_summary = ""
if "summarized_count" not in st.session_state:
    st.session_state.summarized_count = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("ask anything about your document 🎀")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("twinly is thinking... 💭"):
            # Pass everything BEFORE this turn as history, so the model
            # actually gets the rolling-summary context engineering.
            result = ask(
                question,
                history=st.session_state.messages[:-1],
                context_summary=st.session_state.context_summary,
                summarized_count=st.session_state.summarized_count,
            )

        answer = result[0]
        total_tok = result[3]
        cost = result[4]
        latency = result[5]

        # Persist context-engineering state for the next turn.
        st.session_state.context_summary = result[6]
        st.session_state.summarized_count = result[7]

        st.write(answer)
        st.caption(f"⏱️ {latency}s  •  🔢 {total_tok} tokens  •  💸 ${cost:.6f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})