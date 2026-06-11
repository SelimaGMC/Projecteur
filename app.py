"""
app.py — Streamlit interface for the French Cinema RAG system.
"""

import os
import json
import uuid
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cinéma Français",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=Inter:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

.stApp {
    background-color: #0f0f0f;
    color: #ececec;
    font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* ── Hamburger ── */
[data-testid="collapsedControl"] {
    color: #ececec !important;
    background: transparent !important;
    border: none !important;
    top: 12px !important;
    left: 12px !important;
    z-index: 9999 !important;
}
[data-testid="collapsedControl"]:hover {
    background: #1e1e1e !important;
    border-radius: 6px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #222 !important;
}
[data-testid="stSidebar"] * { color: #aaa; }

/* ── Remove Streamlit spacing around sidebar buttons ── */
[data-testid="stSidebar"] [data-testid="stButton"],
[data-testid="stSidebar"] [data-testid="stButton"] > div,
[data-testid="stSidebar"] .btn-conv,
[data-testid="stSidebar"] .btn-conv-active {
    margin: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
}
[data-testid="stSidebar"] .element-container {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button:focus,
[data-testid="stSidebar"] button:active {
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #aaa !important;
    font-size: 0.83rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 1px 10px !important;
    border-radius: 6px !important;
    font-weight: 400 !important;
    transition: background 0.12s, color 0.12s !important;
}
[data-testid="stSidebar"] button:hover {
    background-color: #222 !important;
    border: none !important;
    color: #ececec !important;
}

/* ── New conv and clear buttons — restore normal padding ── */
[data-testid="stSidebar"] .btn-new button {
    color: #ccc !important;
    font-size: 0.88rem !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .btn-clear button {
    color: #666 !important;
    font-size: 0.8rem !important;
    padding: 8px 12px !important;
}
[data-testid="stSidebar"] .btn-clear button:hover {
    color: #e55 !important;
    background-color: #1e1010 !important;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 0.75rem; font-weight: 600;
}
.status-ready   { background:#1a3a1a; color:#4caf50 !important; border:1px solid #4caf50; }
.status-loading { background:#2a2200; color:#d4af37 !important; border:1px solid #d4af37; }
.status-error   { background:#3a0000; color:#f44336 !important; border:1px solid #f44336; }

/* ── Centred content column ── */
.main .block-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 0 1rem 6rem 1rem !important;
}

/* ── Hero block ── */
.hero-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20vh 0 2rem 0;
    gap: 0.8rem;
}
.hero-title-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.logo-img {
    width: 120px !important;
    height: 120px !important;
    filter: none !important;
    flex-shrink: 0;
}
.logo-img.spinning { animation: spin 1.2s linear infinite; }
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

.hero-title {
    font-family: 'Special Elite', serif;
    font-size: 2.1rem;
    font-weight: 400;
    color: #ececec;
    margin: 0;
    line-height: 1;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: #555;
    margin: 0;
    letter-spacing: 0.04em;
}

/* ── Pill buttons ── */
div.pill-btn > div[data-testid="stButton"] > button {
    background: #1a1a1a !important;
    color: #a0a0a0 !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 999px !important;
    padding: 6px 18px !important;
    font-size: 0.82rem !important;
    white-space: nowrap !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.4 !important;
}
div.pill-btn > div[data-testid="stButton"] > button:hover {
    background: #242424 !important;
    border-color: #484848 !important;
    color: #ececec !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
    display: flex;
    justify-content: flex-end;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #2a2a2a !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 10px 16px !important;
    max-width: 80% !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="chatAvatarIcon-user"] {
    display: none !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 14px !important;
    padding: 4px 8px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #555 !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.05) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 0.93rem !important;
    caret-color: #ececec !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #444 !important; }

hr { border-color: #2a2a2a; }

/* ── Interrupted notice ── */
[data-testid="stChatMessage"] em {
    color: #555 !important;
    font-size: 0.88rem !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
WIKI_BOX_OFFICE_URL = (
    "https://fr.wikipedia.org/wiki/"
    "Liste_des_plus_gros_succ%C3%A8s_fran%C3%A7ais_au_box-office_mondial"
)
RETRIEVER_DIR  = "./chroma_db"
LOGO_PATH      = "./logo.png"
HISTORY_FILE   = "./conv_history.json"

EXAMPLE_QUESTIONS = [
    "Qui a réalisé Intouchables ?",
    "Synopsis d'Amélie Poulain",
    "César remportés par Les Choristes",
    "Acteurs de Lucy",
    "Box-office d'Astérix et Obélix",
]


# ─────────────────────────────────────────────────────────────
# CONVERSATION HISTORY  (persisted to JSON)
# ─────────────────────────────────────────────────────────────
def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history: list) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def archive_current_conv() -> None:
    msgs = st.session_state.messages
    if not any(m["role"] == "user" for m in msgs):
        return
    title   = next(m["content"] for m in msgs if m["role"] == "user")[:60]
    conv_id = st.session_state.get("conv_id")
    history = load_history()
    for conv in history:
        if conv["id"] == conv_id:
            conv["messages"] = msgs
            conv["title"]    = title
            save_history(history)
            return
    history.insert(0, {"id": conv_id, "title": title, "messages": msgs})
    save_history(history)


# ─────────────────────────────────────────────────────────────
# RAG INITIALISATION  (cached — runs only once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_chain():
    try:
        import project_secrets as cfg
        os.environ["GOOGLE_API_KEY"] = cfg.GOOGLE_API_KEY
        agent = cfg.CUSTOM_AGENT

        from crawler import load_movies_urls
        urls = []
        if not os.path.exists(RETRIEVER_DIR) or not os.listdir(RETRIEVER_DIR):
            urls, _ = load_movies_urls(WIKI_BOX_OFFICE_URL, agent=agent)

        from retriever import build_knowledge_base
        vectorstore = build_knowledge_base(urls)
        if vectorstore is None:
            return None

        from generator import create_rag_chain
        return create_rag_chain(vectorstore)

    except Exception as exc:
        raise RuntimeError(str(exc)) from exc


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_session_state():
    if "messages"     not in st.session_state:
        st.session_state.messages = []
    if "rag_error"    not in st.session_state:
        st.session_state.rag_error = None
    if "chain_loaded" not in st.session_state:
        st.session_state.chain_loaded = False
    if "conv_id"      not in st.session_state:
        st.session_state.conv_id = str(uuid.uuid4())


# ─────────────────────────────────────────────────────────────
# LEFT SIDEBAR
# Buttons are wrapped in named <div class="btn-*"> so CSS can
# target them reliably without fighting Streamlit's inline styles.
# ─────────────────────────────────────────────────────────────
def render_left_sidebar(chain):
    with st.sidebar:

        # Status badge
        if st.session_state.rag_error:
            st.markdown('<span class="status-badge status-error">⚠ Erreur</span>',
                        unsafe_allow_html=True)
            st.error(st.session_state.rag_error)
        elif chain is not None:
            st.markdown('<span class="status-badge status-ready">● Système prêt</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-loading">◌ Chargement…</span>',
                        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # New conversation
        st.markdown('<div class="btn-new">', unsafe_allow_html=True)
        if st.button("＋  Nouvelle conversation", key="new_conv_btn",
                     use_container_width=True):
            msgs = st.session_state.messages
            if msgs and msgs[-1]["role"] == "user":
                msgs.append({"role": "assistant",
                              "content": "_— Génération interrompue._"})
            archive_current_conv()
            st.session_state.messages = []
            st.session_state.conv_id  = str(uuid.uuid4())
            st.session_state.pop("pending_answer", None)
            st.session_state.pop("pending_q", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Conversation history list
        history    = load_history()
        current_id = st.session_state.conv_id

        if history:
            for conv in history:
                is_active  = conv["id"] == current_id
                div_class  = "btn-conv-active" if is_active else "btn-conv"
                st.markdown(f'<div class="{div_class}">', unsafe_allow_html=True)
                if st.button(conv["title"], key=f"conv-{conv['id']}",
                             use_container_width=True):
                    archive_current_conv()
                    st.session_state.messages = conv["messages"]
                    st.session_state.conv_id  = conv["id"]
                    st.session_state.pop("pending_answer", None)
                    st.session_state.pop("pending_q", None)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="color:#444;font-size:0.8rem;padding:4px 8px;">'
                'Aucune conversation.</p>',
                unsafe_allow_html=True)

        # Clear all — pinned at bottom via CSS margin-top trick
        st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="btn-clear">', unsafe_allow_html=True)
        if st.button("Effacer les conversations", key="clear_all_btn",
                     use_container_width=True):
            save_history([])
            st.session_state.messages = []
            st.session_state.conv_id  = str(uuid.uuid4())
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
def render_hero(spinning: bool = False):
    spin_class = "logo-img spinning" if spinning else "logo-img"
    if os.path.exists(LOGO_PATH):
        import base64
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img_html = f'<img src="data:image/png;base64,{b64}" class="{spin_class}" alt="logo" style="width:140px;height:140px;"/>'
    else:
        img_html = '<span style="font-size:8rem;">🎬</span>'

    st.markdown(f"""
    <div class="hero-wrapper">
        <div class="hero-title-row">
            {img_html}
            <p class="hero-title" style="font-size:4.5rem;font-family:'Special Elite',serif;color:#ececec;margin:0;line-height:1;">Projecteur</p>
        </div>
        <p class="hero-subtitle">Le cinéma français, à la demande.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, (i, q) in zip(cols, enumerate(EXAMPLE_QUESTIONS)):
        with col:
            st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"pill_{i}", use_container_width=True):
                st.session_state["pending_q"] = q
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────────────────────
def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ─────────────────────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────────────────────
def answer_question(chain, question: str):
    # Step 1 — persist user message and rerun so hero disappears instantly
    if not any(m["content"] == question and m["role"] == "user"
               for m in st.session_state.messages[-1:]):
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state["pending_answer"] = question
        st.rerun()

    # Step 2 — generate response
    question_to_answer = st.session_state.pop("pending_answer", None)
    if question_to_answer:
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours…"):
                try:
                    response = chain.invoke(question_to_answer)
                except Exception as exc:
                    response = f"⚠️ Erreur : {exc}"
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        archive_current_conv()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    init_session_state()

    chain   = None
    loading = False
    if not st.session_state.rag_error:
        try:
            chain = load_rag_chain()
            st.session_state.chain_loaded = True
        except RuntimeError as exc:
            st.session_state.rag_error = str(exc)
        loading = not st.session_state.chain_loaded

    render_left_sidebar(chain)

    if not st.session_state.messages:
        render_hero(spinning=loading)

    # Detect Stop-button interruption
    msgs = st.session_state.messages
    if (msgs
            and msgs[-1]["role"] == "user"
            and "pending_answer" not in st.session_state):
        st.session_state.messages.append({
            "role": "assistant",
            "content": "_— Génération interrompue._",
        })

    render_chat_history()

    if "pending_answer" in st.session_state and chain:
        answer_question(chain, st.session_state["pending_answer"])

    pending = st.session_state.pop("pending_q", None)
    if pending and chain:
        answer_question(chain, pending)

    placeholder = (
        "Chargement du système…" if chain is None
        else "Posez votre question…"
    )
    user_input = st.chat_input(placeholder, disabled=(chain is None))
    if user_input:
        answer_question(chain, user_input)


if __name__ == "__main__":
    # Workaround for Python 3.12 + Windows asyncio cleanup bug with Streamlit.
    # Prevents "RuntimeError: Event loop is closed" on Ctrl+C.
    import asyncio
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()