import streamlit as st
import os

st.set_page_config(page_title="Samsung Washing Machine Assistant", page_icon="🫧", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #001a33 100%); color: #f3f4f6; }
    h1, h2, h3 { color: #38bdf8 !important; font-weight: 700 !important; }
    .chat-bubble-user {
        background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.4);
        border-radius: 12px; padding: 12px 16px; margin: 6px 0; text-align: right;
    }
    .chat-bubble-bot {
        background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.3);
        border-radius: 12px; padding: 12px 16px; margin: 6px 0;
    }
    .context-block {
        background: rgba(255,255,255,0.03); border-left: 3px solid #38bdf8;
        border-radius: 4px; padding: 8px 12px; font-size: 0.85rem; color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

st.title("🫧 Samsung Washing Machine Assistant")
st.markdown("A **RAG-powered chatbot** that answers questions about Samsung washing machine cycles and modes.")

# ── Sidebar: API key ──────────────────────────────────────────────────────────
st.sidebar.header("🔑 OpenAI API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password",
    placeholder="sk-proj-...")
st.sidebar.caption("Your key is never stored. [Get a free key](https://platform.openai.com/api-keys)")
st.sidebar.write("---")
st.sidebar.subheader("💡 Sample Questions")
samples = [
    "What is the cycle for DRUM CLEAN?",
    "What should I do for DAILY WASH?",
    "Which cycle is best for wool?",
    "How does the QUICK WASH cycle work?",
    "What is SUPER ECO WASH?",
]
for s in samples:
    if st.sidebar.button(s, key=f"btn_{s[:20]}"):
        st.session_state.pending_query = s

# ── Knowledge base (extracted from Chroma DB) ────────────────────────────────
KNOWLEDGE_BASE = """
How to use the various modes of the washing machine | Samsung LEVANT
Last Update date : Feb 16. 2023

Innovations like Ecobubble™ and QuickDrive™ washing machines can work hard to clean clothes in a fraction of the time and while using less energy.

COTTON: For cottons, bed linens, table linens, underwear, towels, or shirts. The washing time and the rinse count are automatically adjusted according to the load. For cleaning lightly soiled cotton items at a nominal temperature of 20°C. Optimal performance with lower energy for cotton fabrics.

SYNTHETICS: For fabrics that are made of polyester (diolen, trevira), polyamide (perlon, nylon), or the like.

HYGIENE STEAM: Cotton and linen fabric that have been in direct contact with the skin, e.g. underwear. For optimum hygiene results select a temperature of 60°C or above. Steam cycles feature allergy care and bacteria elimination (sterilize).

RINSE+SPIN: Featuring an additional rinse process after applying fabric softener to the laundry.

DRAIN/SPIN: For draining the water inside the drum and running an additional spin process to effectively remove moisture from laundry.

DRUM CLEAN: Cleans the drum by removing dirt and bacteria from it. Perform once every 40 washes with no detergent or bleach applied. A notification appears after every 40 washes. Make sure the drum is empty.

15' QUICK WASH: For lightly soiled items in less than 2.0 kg that you want to wash quickly. Use less than 20 g of detergent. For liquid detergent, use a maximum of 20 ml.

DAILY WASH: For everyday items such as underwear and shirts.

DELICATES: For sheer fabrics, bras, lingerie (silk), and other handwash-only fabrics. For best performance, use liquid detergent.

WOOL: Specific for machine-washable wool for loads less than 2.0 kg. The WOOL cycle features fine pulsating and soaking to protect the wool fibers from shrinkage/distortion. A neutral detergent is recommended.

SUPER ECO WASH: Low-temperature eco bubble cycle helps reduce power consumption.

COLOURS: Featuring additional rinses and reduced spinning to ensure that the laundry is washed gently and rinsed thoroughly.

BEDDING: For bedspreads, bed sheets, bedding covers, etc. For best results, wash only 1 type of bedding and make sure the load weighs less than 2.0 kg.

How to start a cycle:
1. Press the Power button to turn on the washing machine.
2. Turn the Cycle Selector to select a cycle.
3. Change the cycle settings (Temp, Rinse, and Spin) as necessary.
4. To add an option, press Options.
5. Press Start/Pause to begin.

To change the cycle during operation:
1. Press Start/Pause to stop the operation.
2. Select a different cycle.
3. Press Start/Pause again to start the new cycle.
"""

def simple_retrieve(question: str, top_k: int = 3) -> str:
    """Simple keyword-based retrieval fallback."""
    chunks = [c.strip() for c in KNOWLEDGE_BASE.split("\n\n") if c.strip()]
    q_lower = question.lower()
    scored = []
    for chunk in chunks:
        c_lower = chunk.lower()
        score = sum(1 for word in q_lower.split() if len(word) > 3 and word in c_lower)
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    top = [c for _, c in scored[:top_k] if _]
    return "\n\n".join(top) if top else "\n\n".join([c for _, c in scored[:2]])

def answer_with_openai(question: str, context: str, key: str) -> str:
    """Use OpenAI gpt-4o-mini to answer based on context."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        prompt = f"""You are an assistant for Samsung washing machine support.
Use the following context to answer the question concisely (3 sentences max).
If you don't know, say so.

Context:
{context}

Question: {question}
Answer:"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ OpenAI error: {str(e)}"

def answer_local(question: str, context: str) -> str:
    """Simple template-based answer without OpenAI."""
    if not context:
        return "I couldn't find relevant information about that in the manual."
    return f"Based on the Samsung washing machine manual:\n\n{context[:600]}..."

# ── Chat state ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Render chat history ───────────────────────────────────────────────────────
st.write("---")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📄 Retrieved Context (RAG)", expanded=False):
                st.markdown(f'<div class="context-block">{msg["context"]}</div>', unsafe_allow_html=True)

# ── Handle sidebar quick-question ──────────────────────────────────────────────
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query})
    context = simple_retrieve(query)
    if api_key and api_key.startswith("sk-"):
        answer = answer_with_openai(query, context, api_key)
    else:
        answer = answer_local(query, context)
    st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
    st.rerun()

# ── Chat input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about Samsung washing machine cycles..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    context = simple_retrieve(user_input)
    if api_key and api_key.startswith("sk-"):
        answer = answer_with_openai(user_input, context, api_key)
    else:
        answer = answer_local(user_input, context)
        if not api_key:
            answer += "\n\n_ℹ️ Enter your OpenAI API key in the sidebar for smarter AI-powered answers._"
    st.session_state.messages.append({"role": "assistant", "content": answer, "context": context})
    st.rerun()

if not st.session_state.messages:
    st.info("👈 Click a sample question in the sidebar, or type your question below!")
