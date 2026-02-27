"""
LiveKit Voice AI Agent - Latency Optimized
Connects Plivo phone calls to an AI voice bot via LiveKit SIP.

Features:
- Deepgram Nova-2 STT + Aura TTS
- Groq LLM (fast inference) with function calling, GPT-4o-mini fallback
- Per-turn latency tracking (saved to DB)
- PostgreSQL call logging with AI summaries
- Correct <User> SIP tag for Plivo forwarding
"""

import os
import sys
import json
import asyncio
import signal
import subprocess
import time
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, openai, silero

load_dotenv(dotenv_path="../.env")
load_dotenv()

# ============ CONFIGURATION ============

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
LIVEKIT_SIP_URI = os.getenv("LIVEKIT_SIP_URI", "")
PORT = int(os.getenv("PORT", 8080))


# ============ DATABASE ============

def init_database():
    if not DATABASE_URL:
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS call_logs (
                id SERIAL PRIMARY KEY,
                call_id VARCHAR(100),
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds FLOAT,
                summary TEXT,
                transcript JSONB,
                message_count INTEGER,
                avg_latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Add columns if table already exists without them
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS avg_latency_ms FLOAT;
                ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS latency_metrics JSONB;
            EXCEPTION WHEN others THEN NULL;
            END $$;
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("PostgreSQL database initialized!", flush=True)
        return True
    except Exception as e:
        print(f"Database init error: {e}", flush=True)
        return False


def save_call_to_db(call_id, start_time, duration, summary, transcript, count, avg_latency_ms=None, latency_metrics=None):
    if not DATABASE_URL:
        return False
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO call_logs (call_id, start_time, end_time, duration_seconds, summary, transcript, message_count, avg_latency_ms, latency_metrics) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (call_id, start_time, datetime.now(), duration, summary, json.dumps(transcript), count, avg_latency_ms, json.dumps(latency_metrics) if latency_metrics else None)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"Call saved to PostgreSQL: {call_id}", flush=True)
        return True
    except Exception as e:
        print(f"DB Save Error: {e}", flush=True)
        return False


def get_call_logs_from_db(limit=10):
    if not DATABASE_URL:
        return []
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT call_id, start_time, duration_seconds, summary, message_count, avg_latency_ms, created_at, latency_metrics
            FROM call_logs ORDER BY created_at DESC LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{
            "call_id": r[0],
            "start_time": r[1].isoformat() if r[1] else None,
            "duration_seconds": r[2],
            "summary": r[3],
            "message_count": r[4],
            "avg_latency_ms": r[5],
            "created_at": r[6].isoformat() if r[6] else None,
            "latency_metrics": r[7],
        } for r in rows]
    except Exception as e:
        print(f"DB Read Error: {e}", flush=True)
        return []


# ============ KNOWLEDGE BASE ============

KNOWLEDGE_BASE = {
    "company_overview": {
        "keywords": ["company", "techcorp", "about", "what do you do", "services", "what is", "tell me about", "who are you"],
        "answer": "TechCorp is a technology company offering cloud storage, analytics, and collaboration tools for individuals and businesses. We have Basic, Pro, and Enterprise plans."
    },
    "business_hours": {
        "keywords": ["hours", "open", "close", "timing", "when are you open", "working hours", "schedule"],
        "answer": "We're open Monday through Friday from 9 AM to 6 PM, and Saturday from 10 AM to 4 PM. We're closed on Sundays."
    },
    "location": {
        "keywords": ["address", "location", "where", "office", "visit", "directions"],
        "answer": "Our office is located at 123 Tech Street, Koramangala, Bangalore. We're near the Metro station."
    },
    "contact": {
        "keywords": ["contact", "email", "phone", "reach", "call", "support email"],
        "answer": "You can reach us by email at support@techcorp.com or call our support line at 1800-123-4567."
    },
    "pricing": {
        "keywords": ["price", "cost", "pricing", "plans", "subscription", "how much", "rates", "packages"],
        "answer": "We have three plans: Basic at $9.99 per month with 10GB storage, Pro at $19.99 per month with 100GB storage and priority support, and Enterprise with custom pricing for unlimited storage and dedicated support."
    },
    "basic_plan": {
        "keywords": ["basic plan", "starter", "cheapest", "entry level", "basic"],
        "answer": "Our Basic plan is $9.99 per month. It includes 10GB storage, email support, and access to core features. Great for individuals."
    },
    "pro_plan": {
        "keywords": ["pro plan", "professional", "premium", "pro"],
        "answer": "Our Pro plan is $19.99 per month. It includes 100GB storage, priority support, advanced analytics, and API access. Perfect for small teams."
    },
    "enterprise_plan": {
        "keywords": ["enterprise", "business", "corporate", "custom", "unlimited"],
        "answer": "Our Enterprise plan has custom pricing based on your needs. It includes unlimited storage, dedicated support, custom integrations, and SLA guarantees. Contact sales for a quote."
    },
    "return_policy": {
        "keywords": ["return", "refund", "money back", "cancel", "cancellation"],
        "answer": "We offer a 30-day money-back guarantee. If you're not satisfied, contact support within 30 days of purchase for a full refund. No questions asked."
    },
    "shipping": {
        "keywords": ["shipping", "delivery", "how long", "ship time", "when will i get"],
        "answer": "Standard shipping takes 3 to 5 business days. Express shipping is available for 1 to 2 day delivery at an additional cost."
    },
    "warranty": {
        "keywords": ["warranty", "guarantee", "broken", "defect", "repair"],
        "answer": "All our products come with a 1-year warranty covering manufacturing defects. Extended warranty options are available at checkout."
    },
    "reset_password": {
        "keywords": ["password", "reset", "forgot", "login", "can't login", "locked out", "sign in"],
        "answer": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email and we'll send you a reset link. The link expires in 24 hours."
    },
    "account_setup": {
        "keywords": ["setup", "get started", "create account", "sign up", "register", "new account"],
        "answer": "To create an account, visit our website and click 'Sign Up'. Enter your email, create a password, and verify your email. Setup takes less than 2 minutes."
    },
    "technical_issues": {
        "keywords": ["not working", "bug", "error", "problem", "issue", "broken", "help", "trouble", "fix"],
        "answer": "I'm sorry you're experiencing issues. For technical problems, please email support@techcorp.com with details, or call our tech support at 1800-123-4567 option 2."
    },
    "order_status": {
        "keywords": ["order status", "track", "where is my order", "order number", "tracking", "my order"],
        "answer": "To check your order status, visit our website and go to 'My Orders', or give me your order number and I can look it up for you."
    },
    "bulk_discount": {
        "keywords": ["bulk", "discount", "wholesale", "volume", "many licenses", "team pricing"],
        "answer": "Yes, we offer bulk discounts for orders of 10 or more licenses. Contact our sales team at sales@techcorp.com for a custom quote."
    },
    "payment_methods": {
        "keywords": ["payment", "pay", "credit card", "paypal", "how to pay", "billing"],
        "answer": "We accept all major credit cards, PayPal, and bank transfers for annual plans. Enterprise customers can also pay by invoice."
    },
    "features": {
        "keywords": ["features", "what can", "capabilities", "functionality", "tools", "integrations"],
        "answer": "TechCorp offers cloud storage, team collaboration, advanced analytics dashboards, API access, custom integrations, and 24/7 support depending on your plan."
    },
    "security": {
        "keywords": ["security", "secure", "encryption", "data", "privacy", "safe", "gdpr", "compliance"],
        "answer": "We use AES-256 encryption for all data at rest and TLS 1.3 for data in transit. We're SOC 2 Type II certified and GDPR compliant."
    },
    "free_trial": {
        "keywords": ["free trial", "trial", "try", "demo", "test", "free"],
        "answer": "Yes! All our plans come with a 14-day free trial. No credit card required to start. You can upgrade or cancel anytime during the trial."
    },
}


def search_knowledge_base(query: str) -> str:
    """Search the knowledge base using keyword matching."""
    query_lower = query.lower()
    best_match = None
    best_score = 0

    for topic, data in KNOWLEDGE_BASE.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in query_lower:
                score += len(keyword)
        if score > best_score:
            best_score = score
            best_match = data["answer"]

    return best_match


# ============ AGENT TOOLS ============

# Global reference for hangup — set per call in entrypoint
_call_disconnect_event = None


@llm.function_tool
async def search_knowledge(query: str) -> str:
    """Search company knowledge base for information about products, services, pricing, policies, hours, support, features, security, etc. Use this whenever the user asks a question about TechCorp."""
    result = search_knowledge_base(query)
    if result:
        return result
    return "I don't have specific information about that in our knowledge base. Would you like me to schedule a callback with a team member who can help?"


@llm.function_tool
async def lookup_order(order_id: str) -> str:
    """Look up order status by order ID."""
    return f"Order {order_id} is in transit, arriving in 2-3 business days."


@llm.function_tool
async def schedule_callback(department: str, preferred_time: str = "as soon as possible") -> str:
    """Schedule a callback from sales, support, or billing."""
    return f"I've scheduled a {department} callback for {preferred_time}. They'll call you back."


@llm.function_tool
async def end_call(reason: str = "caller said goodbye") -> str:
    """End the phone call. Use this ONLY when the caller clearly wants to end the call — they said goodbye, bye, that's all, I'm done, etc. Say a brief farewell before calling this."""
    global _call_disconnect_event
    print(f"[FUNCTION] end_call triggered (reason: {reason})", flush=True)
    if _call_disconnect_event is not None:
        # Small delay so the farewell TTS can play first
        await asyncio.sleep(3)
        _call_disconnect_event.set()
    return "Call ended."


# ============ AGENT ============

class TechCorpReceptionist(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a friendly and professional AI phone receptionist for TechCorp.

GREETING: Start every call with "Hello, thank you for calling TechCorp! How can I help you today?"
If the user says "Hello?" or "Hi" first, respond warmly: "Hi there! Welcome to TechCorp. How can I help you today?" Do NOT treat "Hello?" as an error.

CAPABILITIES - Use these tools:
- search_knowledge: For ANY question about TechCorp — services, products, pricing, hours, location, policies, features, security, free trial, etc. ALWAYS try this first before saying you don't know.
- lookup_order: When customer asks about an order status (ask for order ID if not provided).
- schedule_callback: When customer wants a callback from sales, support, or billing.

ABOUT TECHCORP: TechCorp offers cloud storage, analytics, and collaboration tools. We have Basic ($9.99/mo), Pro ($19.99/mo), and Enterprise (custom) plans. 14-day free trial available. When asked about services or what we do, use search_knowledge to give detailed info.

CONVERSATION STYLE:
- Keep responses to 1-2 sentences. This is a phone call.
- Be natural and conversational. No bullet points, lists, or formatting.
- Remember what was discussed earlier ("As I mentioned earlier...").
- After answering, occasionally ask "Is there anything else I can help with?"

ENDING CALLS:
- When the caller says goodbye, bye, that's all, I'm done, etc.: Say "Thank you for calling TechCorp. Have a great day!" then call the end_call tool.
- ALWAYS call end_call after saying goodbye. This hangs up the phone.
- Do NOT call end_call for simple "thank you" or "thanks" — those are just acknowledgments, not goodbyes.
- ALWAYS ask if the caller wants to end the call before calling end_call.

IMPORTANT: NEVER say you can't help or don't have info without trying search_knowledge first. Always attempt the tool before falling back.""",
            tools=[search_knowledge, lookup_order, schedule_callback, end_call],
        )


# ============ ENTRYPOINT ============

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    global _call_disconnect_event

    call_id = "unknown"
    call_start = datetime.now()
    transcript = []
    session = None  # keep reference for chat_ctx extraction in finally

    # Per-component metrics collected from SDK's metrics_collected event
    turn_metrics_list = []       # completed turns [{stt_ms, llm_first_token_ms, ...}]
    current_turn = {}            # in-progress turn being built from metrics events
    llm_provider_name = [None]   # track which LLM is in use

    # Disconnect event — shared with end_call tool
    disconnect_event = asyncio.Event()
    _call_disconnect_event = disconnect_event

    try:
        print("[Agent] Entrypoint called, connecting...", flush=True)
        await ctx.connect()

        participant = await ctx.wait_for_participant()
        call_id = participant.identity or "unknown"
        print(f"[Agent] Participant joined: {call_id}", flush=True)

        # Use Groq for fast LLM inference (~300ms first token), fall back to GPT-4o-mini
        active_llm = None
        if GROQ_API_KEY:
            # Try with_groq() class method (livekit-plugins-openai v1.4+)
            try:
                active_llm = openai.LLM.with_groq(
                    model="llama-3.3-70b-versatile",
                    api_key=GROQ_API_KEY,
                )
                llm_provider_name[0] = "groq"
                print("[Agent] LLM: Groq llama-3.3-70b-versatile (via with_groq)", flush=True)
            except (AttributeError, TypeError) as e:
                print(f"[Agent] with_groq() not available ({e}), trying base_url...", flush=True)
                try:
                    active_llm = openai.LLM(
                        model="llama-3.3-70b-versatile",
                        base_url="https://api.groq.com/openai/v1",
                        api_key=GROQ_API_KEY,
                    )
                    llm_provider_name[0] = "groq"
                    print("[Agent] LLM: Groq llama-3.3-70b-versatile (via base_url)", flush=True)
                except Exception as e2:
                    print(f"[Agent] Groq base_url failed ({e2}), falling back to GPT-4o-mini", flush=True)
                    active_llm = None
        if active_llm is None:
            active_llm = openai.LLM(model="gpt-4o-mini")
            llm_provider_name[0] = "openai"
            print("[Agent] LLM: GPT-4o-mini", flush=True)

        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(model="nova-2"),
            llm=active_llm,
            tts=deepgram.TTS(model="aura-asteria-en"),
        )

        # ---- Deduplication helper for AI response capture ----
        _last_ai_text = [None]

        def _record_ai_response(text):
            """Add AI response to transcript with deduplication."""
            if not text or not text.strip():
                return
            text = text.strip()
            if text == _last_ai_text[0]:
                return  # duplicate
            _last_ai_text[0] = text
            print(f"[AI] {text}", flush=True)
            transcript.append(("AI", text))

        # ---- Track user speech transcripts ----
        @session.on("user_input_transcribed")
        def on_user(event):
            try:
                # Skip interim/partial results — only log final transcripts
                is_final = getattr(event, "is_final", True)
                if not is_final:
                    return
                text = getattr(event, "transcript", None) or getattr(event, "text", None)
                if text and isinstance(text, str) and text.strip():
                    print(f"[User] {text.strip()}", flush=True)
                    transcript.append(("User", text.strip()))
            except Exception as e:
                print(f"[Event] user_input_transcribed error: {e}", flush=True)

        # ---- Track AI + User speech via conversation_item_added (v1.x primary event) ----
        # Note: agent_speech_committed does NOT exist in v1.x — it was a v0.x event
        @session.on("conversation_item_added")
        def on_conversation_item(event):
            try:
                item = getattr(event, "item", event)
                role = getattr(item, "role", None)
                if role not in ("assistant", "agent"):
                    return
                # v1.4 ChatMessage has .text_content property that returns concatenated text
                text = getattr(item, "text_content", None)
                if not text:
                    # Fallback: try content list
                    content = getattr(item, "content", None)
                    if isinstance(content, str) and content.strip():
                        text = content.strip()
                    elif isinstance(content, list):
                        parts = []
                        for cp in content:
                            t = getattr(cp, "text", None) or (cp if isinstance(cp, str) else None)
                            if t and isinstance(t, str):
                                parts.append(t.strip())
                        text = " ".join(parts) if parts else None
                if text:
                    _record_ai_response(text)
                else:
                    # Debug: log what attributes the item actually has so we can diagnose
                    attrs = {a: type(getattr(item, a, None)).__name__ for a in dir(item) if not a.startswith('_')}
                    print(f"[Event] conversation_item_added: assistant item with no text. Attrs: {attrs}", flush=True)
            except Exception as e:
                print(f"[Event] conversation_item_added error: {e}", flush=True)

        # ---- Collect per-component metrics from SDK (STT, LLM, TTS) ----
        @session.on("metrics_collected")
        def on_metrics(event):
            try:
                metrics = getattr(event, "metrics", event)
                metric_type = type(metrics).__name__

                if "STT" in metric_type:
                    # New user turn — save previous turn if it was complete (had LLM+TTS)
                    if current_turn.get("llm_first_token_ms") is not None:
                        turn_metrics_list.append(dict(current_turn))
                        current_turn.clear()
                    elif current_turn:
                        # Previous turn was STT-only (no LLM/TTS) — discard it
                        current_turn.clear()

                    stt_ms = (getattr(metrics, "duration", 0) or 0) * 1000
                    current_turn["stt_ms"] = round(stt_ms, 1)
                    print(f"[Metrics] STT: {stt_ms:.0f}ms", flush=True)

                elif "LLM" in metric_type:
                    ttft = (getattr(metrics, "ttft", 0) or 0) * 1000
                    llm_dur = (getattr(metrics, "duration", 0) or 0) * 1000
                    tokens_per_sec = getattr(metrics, "tokens_per_second", None)
                    current_turn["llm_first_token_ms"] = round(ttft, 1)
                    current_turn["llm_total_ms"] = round(llm_dur, 1)
                    if tokens_per_sec:
                        current_turn["tokens_per_second"] = round(tokens_per_sec, 1)
                    # Track provider
                    if llm_provider_name[0]:
                        current_turn["llm_provider"] = llm_provider_name[0]
                    print(f"[Metrics] LLM: TTFT={ttft:.0f}ms, total={llm_dur:.0f}ms", flush=True)

                elif "TTS" in metric_type:
                    ttfb = (getattr(metrics, "ttfb", 0) or 0) * 1000
                    tts_dur = (getattr(metrics, "duration", 0) or 0) * 1000
                    current_turn["tts_first_chunk_ms"] = round(ttfb, 1)
                    current_turn["tts_total_ms"] = round(tts_dur, 1)

                    # Use transcription_delay as effective STT latency when stt_ms is 0
                    # (Deepgram streaming STT reports duration=0 since it processes incrementally)
                    stt = current_turn.get("stt_ms", 0)
                    if stt == 0:
                        stt = current_turn.get("transcription_delay_ms", 0)
                        if stt > 0:
                            current_turn["stt_ms"] = stt

                    llm_ttft = current_turn.get("llm_first_token_ms", 0)
                    current_turn["time_to_first_audio_ms"] = round(stt + llm_ttft + ttfb, 1)
                    current_turn["total_ms"] = round(stt + current_turn.get("llm_total_ms", 0) + tts_dur, 1)
                    print(f"[Metrics] TTS: TTFB={ttfb:.0f}ms, total={tts_dur:.0f}ms | Turn TTFA={current_turn['time_to_first_audio_ms']:.0f}ms", flush=True)

                elif "EOU" in metric_type:
                    eou_delay = (getattr(metrics, "end_of_utterance_delay", 0) or 0) * 1000
                    transcription_delay = (getattr(metrics, "transcription_delay", 0) or 0) * 1000
                    if eou_delay > 0:
                        current_turn["eou_delay_ms"] = round(eou_delay, 1)
                    if transcription_delay > 0:
                        current_turn["transcription_delay_ms"] = round(transcription_delay, 1)
                        # Backfill stt_ms if it was 0 (streaming STT reports no duration)
                        if current_turn.get("stt_ms", 0) == 0:
                            current_turn["stt_ms"] = round(transcription_delay, 1)
                    print(f"[Metrics] EOU: delay={eou_delay:.0f}ms, transcription={transcription_delay:.0f}ms", flush=True)

                elif "VAD" in metric_type:
                    pass  # VAD metrics are noisy, suppress

                else:
                    print(f"[Metrics] {metric_type}", flush=True)

            except Exception as e:
                print(f"[Metrics] Error processing {type(getattr(event, 'metrics', event)).__name__}: {e}", flush=True)

        await session.start(agent=TechCorpReceptionist(), room=ctx.room)
        print("[Agent] Session started!", flush=True)

        # Disconnect on participant leave
        def on_participant_disconnected(participant):
            print(f"[Agent] Participant disconnected: {participant.identity}", flush=True)
            disconnect_event.set()

        ctx.room.on("participant_disconnected", on_participant_disconnected)

        async def on_shutdown():
            disconnect_event.set()

        ctx.add_shutdown_callback(on_shutdown)

        # Wait for disconnect (from participant leaving, end_call tool, or shutdown)
        await disconnect_event.wait()
        print("[Agent] Disconnect event received", flush=True)

    except (asyncio.CancelledError, Exception) as e:
        if not isinstance(e, asyncio.CancelledError):
            import traceback
            print(f"[Agent] ENTRYPOINT ERROR: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

    finally:
        _call_disconnect_event = None

        # Finalize any in-progress turn metrics
        if current_turn:
            turn_metrics_list.append(dict(current_turn))
            current_turn.clear()

        # Fallback: extract transcript from session.history (v1.4 ChatContext)
        if session is not None:
            try:
                history = getattr(session, "history", None)
                if history is not None:
                    # .messages() returns only ChatMessage items (filters out function calls etc.)
                    messages_fn = getattr(history, "messages", None)
                    items = messages_fn() if callable(messages_fn) else (getattr(history, "items", None) or [])
                    session_transcript = []
                    for msg in items:
                        role = getattr(msg, "role", None)
                        if role == "system":
                            continue
                        # v1.4 ChatMessage has .text_content property
                        text = getattr(msg, "text_content", None)
                        if not text:
                            content = getattr(msg, "content", None)
                            if isinstance(content, str) and content.strip():
                                text = content.strip()
                        if text and text.strip():
                            speaker = "AI" if role in ("assistant", "agent") else "User"
                            session_transcript.append((speaker, text.strip()))

                    ai_from_events = sum(1 for t in transcript if t[0] == "AI")
                    ai_from_session = sum(1 for t in session_transcript if t[0] == "AI")
                    print(f"[Agent] Transcript comparison: events={len(transcript)} ({ai_from_events} AI), history={len(session_transcript)} ({ai_from_session} AI)", flush=True)

                    # Use session history if it captured more AI responses
                    if ai_from_session > ai_from_events and len(session_transcript) > 0:
                        print(f"[Agent] Using session.history transcript (has {ai_from_session} AI responses vs {ai_from_events} from events)", flush=True)
                        transcript = session_transcript
                else:
                    print("[Agent] session.history not available", flush=True)
            except Exception as e:
                print(f"[Agent] chat_ctx extraction failed: {e}", flush=True)

        print(f"[Agent] Saving call data ({len(transcript)} messages, {len(turn_metrics_list)} turns with metrics)...", flush=True)
        _save_call_sync(call_id, call_start, transcript, turn_metrics_list)


def _save_call_sync(call_id, call_start, transcript, turn_metrics_list):
    """Save call data synchronously — guaranteed to complete before process exits."""
    if not transcript:
        print("[Agent] No transcript to save", flush=True)
        return

    duration = (datetime.now() - call_start).total_seconds()

    # Build latency_metrics in Pipecat-compatible format from SDK metrics
    # Filter out incomplete turns (STT-only with no LLM/TTS — not real conversational turns)
    latency_metrics = None
    avg_latency = None
    complete_turns = [tm for tm in turn_metrics_list if tm.get("llm_first_token_ms") is not None or tm.get("tts_first_chunk_ms") is not None]
    if complete_turns:
        turns = []
        for i, tm in enumerate(complete_turns):
            turn_data = {"turn": i + 1}
            for key in ["stt_ms", "llm_first_token_ms", "llm_total_ms", "tts_first_chunk_ms",
                         "tts_total_ms", "time_to_first_audio_ms", "total_ms",
                         "llm_provider", "tokens_per_second", "eou_delay_ms", "transcription_delay_ms"]:
                if key in tm:
                    turn_data[key] = tm[key]
            turns.append(turn_data)

        # Compute averages for numeric fields
        averages = {}
        for key in ["stt_ms", "llm_first_token_ms", "llm_total_ms", "tts_first_chunk_ms",
                     "tts_total_ms", "time_to_first_audio_ms", "total_ms"]:
            values = [t.get(key) for t in complete_turns if t.get(key) is not None]
            if values:
                averages[key] = round(sum(values) / len(values), 1)

        latency_metrics = {
            "turns": turns,
            "averages": averages,
            "total_turns": len(turns),
        }
        avg_latency = averages.get("time_to_first_audio_ms") or averages.get("total_ms")

    print(f"\n{'='*50}", flush=True)
    print(f"CALL ENDED: {call_id}", flush=True)
    print(f"Duration: {duration:.1f}s | Messages: {len(transcript)} | Turns: {len(complete_turns)}", flush=True)
    if complete_turns:
        for i, tm in enumerate(complete_turns):
            parts = []
            if "stt_ms" in tm: parts.append(f"STT={tm['stt_ms']:.0f}ms")
            if "llm_first_token_ms" in tm: parts.append(f"LLM-TTFT={tm['llm_first_token_ms']:.0f}ms")
            if "llm_total_ms" in tm: parts.append(f"LLM={tm['llm_total_ms']:.0f}ms")
            if "tts_first_chunk_ms" in tm: parts.append(f"TTS-TTFB={tm['tts_first_chunk_ms']:.0f}ms")
            if "tts_total_ms" in tm: parts.append(f"TTS={tm['tts_total_ms']:.0f}ms")
            if "time_to_first_audio_ms" in tm: parts.append(f"TTFA={tm['time_to_first_audio_ms']:.0f}ms")
            print(f"  Turn {i+1}: {' | '.join(parts)}", flush=True)
        if avg_latency:
            print(f"  Avg TTFA: {avg_latency:.0f}ms", flush=True)

    # Generate summary synchronously using openai SDK
    summary = "Call ended before summary could be generated."
    if OPENAI_API_KEY and transcript:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            text = "\n".join([f"{r}: {t}" for r, t in transcript])
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Brief 1-2 sentence summary of this customer service call."},
                    {"role": "user", "content": f"Transcript:\n{text}"}
                ],
                max_tokens=100,
                timeout=5,
            )
            summary = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary generation error: {e}", flush=True)

    print(f"Summary: {summary}", flush=True)
    saved = save_call_to_db(call_id, call_start, duration, summary, transcript, len(transcript), avg_latency, latency_metrics)
    print(f"Saved to DB: {'Yes' if saved else 'No'}", flush=True)
    print(f"{'='*50}\n", flush=True)


# ============ FASTAPI SERVER ============

def create_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    app = FastAPI()

    @app.get("/")
    async def health():
        return {
            "status": "ok",
            "service": "TechCorp Voice AI (LiveKit)",
            "stt": "deepgram-nova-3",
            "tts": "deepgram-aura",
            "llm": "groq-llama-3.3-70b" if GROQ_API_KEY else "gpt-4o-mini",
            "sip_configured": bool(LIVEKIT_SIP_URI),
        }

    @app.get("/call-logs")
    async def call_logs():
        logs = get_call_logs_from_db(limit=20)
        return {
            "call_logs": logs,
            "total": len(logs),
            "source": "postgresql" if logs else "none",
        }

    @app.api_route("/sip-forward", methods=["GET", "POST"])
    async def sip_forward(request: Request):
        form = await request.form()
        caller = form.get("From", "unknown")
        to = form.get("To", "unknown")

        if not LIVEKIT_SIP_URI:
            return Response(
                content='<Response><Speak>SIP not configured.</Speak></Response>',
                media_type="application/xml",
            )

        if to != "unknown" and not to.startswith("+"):
            to = "+" + to

        sip_host = LIVEKIT_SIP_URI.replace("sip:", "").strip()
        sip_uri = f"sip:{to}@{sip_host}"
        print(f"[SIP] Forwarding {caller} to {sip_uri}", flush=True)

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial timeout="30" callerId="{caller}">
        <User>{sip_uri}</User>
    </Dial>
    <Speak>Sorry, we could not connect you.</Speak>
</Response>"""
        return Response(content=xml, media_type="application/xml")

    @app.api_route("/hangup", methods=["GET", "POST"])
    async def hangup(request: Request):
        form = await request.form()
        print(f"[SIP] Call ended - From: {form.get('From', '?')}, Duration: {form.get('Duration', '?')}s", flush=True)
        return {"status": "ok"}

    return app


# ============ MAIN ============

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "dev":
        if DATABASE_URL:
            init_database()
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, agent_name="Livekit-demo"))

    elif mode == "serve":
        if DATABASE_URL:
            init_database()
        import uvicorn
        uvicorn.run(create_app(), host="0.0.0.0", port=PORT)

    else:
        script = os.path.abspath(__file__)
        procs = [
            subprocess.Popen([sys.executable, script, "dev"]),
            subprocess.Popen([sys.executable, script, "serve"]),
        ]

        def stop(s, f):
            for p in procs:
                p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        while True:
            for p in procs:
                if p.poll() is not None:
                    stop(None, None)
            time.sleep(1)
