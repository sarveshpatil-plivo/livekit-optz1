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
async def end_call() -> str:
    """End the phone call. Use this ONLY when the caller clearly wants to end the call — they said goodbye, bye, that's all, I'm done, etc. Say a brief farewell before calling this."""
    global _call_disconnect_event
    print("[FUNCTION] end_call triggered", flush=True)
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
    latency_values = []
    last_user_speech_time = [None]

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
        if GROQ_API_KEY:
            active_llm = openai.LLM(
                model="llama-3.3-70b-versatile",
                base_url="https://api.groq.com/openai/v1",
                api_key=GROQ_API_KEY,
            )
            print("[Agent] LLM: Groq llama-3.3-70b-versatile", flush=True)
        else:
            active_llm = openai.LLM(model="gpt-4o-mini")
            print("[Agent] LLM: GPT-4o-mini (set GROQ_API_KEY for faster inference)", flush=True)

        session = AgentSession(
            vad=ctx.proc.userdata["vad"],
            stt=deepgram.STT(model="nova-2"),
            llm=active_llm,
            tts=deepgram.TTS(model="aura-asteria-en"),
        )

        # Track user speech for latency measurement
        @session.on("user_input_transcribed")
        def on_user(event):
            try:
                # Skip interim/partial results — only log final transcripts
                is_final = getattr(event, "is_final", True)
                if not is_final:
                    return

                text = getattr(event, "transcript", None) or getattr(event, "text", None)
                if text and isinstance(text, str) and text.strip():
                    last_user_speech_time[0] = time.time()
                    print(f"[User] {text.strip()}", flush=True)
                    transcript.append(("User", text.strip()))
            except Exception as e:
                print(f"[Event] user_input_transcribed error: {e}", flush=True)

        # Helper: record AI response text + latency
        def _record_ai_response(text, source="agent"):
            if not text or not text.strip():
                return False
            text = text.strip()
            # Deduplicate — skip if already in recent transcript
            if any(t[1] == text for t in transcript[-5:] if t[0] == "AI"):
                return False
            if last_user_speech_time[0] is not None:
                latency_ms = (time.time() - last_user_speech_time[0]) * 1000
                latency_values.append(latency_ms)
                print(f"[AI] {text}  (latency: {latency_ms:.0f}ms) [{source}]", flush=True)
                last_user_speech_time[0] = None
            else:
                print(f"[AI] {text} [{source}]", flush=True)
            transcript.append(("AI", text))
            return True

        # Helper: extract text from an event/item trying multiple attribute patterns
        def _extract_text(obj):
            # Direct string attributes
            for attr in ["content", "text", "transcript", "message", "speech"]:
                val = getattr(obj, attr, None)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
            # Nested content list (e.g., list of ContentPart objects)
            content = getattr(obj, "content", None)
            if isinstance(content, list):
                for cp in content:
                    for attr in ["text", "content"]:
                        t = getattr(cp, attr, None)
                        if t and isinstance(t, str) and t.strip():
                            return t.strip()
            # Try .item sub-object
            item = getattr(obj, "item", None)
            if item and item is not obj:
                return _extract_text(item)
            return None

        # Primary: agent_speech_committed
        @session.on("agent_speech_committed")
        def on_agent_speech(event):
            try:
                text = _extract_text(event)
                if text:
                    _record_ai_response(text, source="speech_committed")
                else:
                    # Log full event structure for debugging
                    attrs = {}
                    for a in dir(event):
                        if not a.startswith('_'):
                            try:
                                v = getattr(event, a)
                                if not callable(v):
                                    attrs[a] = f"{type(v).__name__}={repr(v)[:80]}"
                            except:
                                pass
                    print(f"[DEBUG] agent_speech_committed no text: {type(event).__name__} {attrs}", flush=True)
            except Exception as e:
                print(f"[Event] agent_speech_committed error: {e}", flush=True)

        # Fallback: conversation_item_added
        @session.on("conversation_item_added")
        def on_conversation_item(event):
            try:
                item = getattr(event, "item", event)
                role = getattr(item, "role", None)
                # Accept both "assistant" and "agent" roles
                if role not in ("assistant", "agent"):
                    return
                text = _extract_text(item)
                if text:
                    _record_ai_response(text, source="conv_item")
            except Exception as e:
                print(f"[Event] conversation_item_added error: {e}", flush=True)

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
        print(f"[Agent] Saving call data ({len(transcript)} messages, {len(latency_values)} latency samples)...", flush=True)
        _save_call_sync(call_id, call_start, transcript, latency_values)


def _save_call_sync(call_id, call_start, transcript, latency_values):
    """Save call data synchronously — guaranteed to complete before process exits."""
    if not transcript:
        print("[Agent] No transcript to save", flush=True)
        return

    avg_latency = round(sum(latency_values) / len(latency_values), 1) if latency_values else None
    duration = (datetime.now() - call_start).total_seconds()

    # Build per-turn latency_metrics (similar to Pipecat format)
    latency_metrics = None
    if latency_values:
        turns = []
        for i, lat in enumerate(latency_values):
            turns.append({
                "turn": i + 1,
                "total_ms": round(lat, 1),
                "time_to_first_audio_ms": round(lat, 1),  # best approximation with LiveKit SDK
            })
        latency_metrics = {
            "turns": turns,
            "averages": {
                "total_ms": round(sum(latency_values) / len(latency_values), 1),
                "time_to_first_audio_ms": round(sum(latency_values) / len(latency_values), 1),
            },
            "total_turns": len(latency_values),
            "note": "LiveKit SDK abstracts STT/LLM/TTS pipeline; total_ms = time from user speech end to AI speech start"
        }

    print(f"\n{'='*50}", flush=True)
    print(f"CALL ENDED: {call_id}", flush=True)
    print(f"Duration: {duration:.1f}s | Messages: {len(transcript)}", flush=True)
    if avg_latency:
        print(f"Avg latency: {avg_latency}ms", flush=True)
        for i, lat in enumerate(latency_values):
            print(f"  Turn {i+1}: {lat:.0f}ms", flush=True)

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
            "stt": "deepgram-nova-2",
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
