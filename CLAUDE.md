# CLAUDE.md

## Project Overview

Latency-optimized LiveKit Voice AI Agent. Connects Plivo phone calls to an AI voice bot via LiveKit SIP with Deepgram Nova-2 STT, Deepgram Aura TTS, and OpenAI GPT-4o-mini.

## Architecture

Two-process model (because `cli.run_app()` is blocking and needs main-thread signals):

- `python livekit_agent.py` (default) spawns two subprocesses:
  - `python livekit_agent.py dev` — LiveKit agent worker
  - `python livekit_agent.py serve` — FastAPI server (health + SIP forward)
- Both share PostgreSQL for call logs.

Call flow: Phone → Plivo → `/sip-forward` (FastAPI, `<User>` SIP tag) → LiveKit Cloud SIP Trunk → LiveKit Agent (Deepgram STT → GPT-4o-mini → Deepgram TTS) → Audio back to caller.

## Files

```
livekit_agent.py    # Entire application (agent + FastAPI + process manager)
Dockerfile          # Python 3.11-slim with gcc, libffi-dev, curl
railway.toml        # Railway deployment config
requirements.txt    # Python dependencies
.env                # API keys (not in git)
```

## Key Code Sections (livekit_agent.py)

| Lines | Section | Description |
|-------|---------|-------------|
| 1-46 | Imports & config | Env vars, LiveKit SDK imports, dotenv |
| 48-134 | Database | `init_database()`, `save_call_to_db()`, `get_call_logs_from_db()` |
| 137-238 | Knowledge base | 20-topic `KNOWLEDGE_BASE` dict + keyword-scored `search_knowledge_base()` |
| 241-278 | Function tools | `search_knowledge`, `lookup_order`, `schedule_callback`, `end_call` |
| 280-311 | Agent class | `TechCorpReceptionist(Agent)` with system prompt and all 4 tools |
| 313-507 | Entrypoint | `prewarm()`, `entrypoint()`, event handlers, latency tracking, `_save_call_sync()` |
| 509-571 | FastAPI | Health, call-logs, sip-forward, hangup endpoints |
| 574-609 | Main | Three-mode entry point (dev / serve / both) |

## Endpoints (FastAPI — serve mode)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check (returns STT/TTS/LLM versions) |
| `/call-logs` | GET | Call history with AI summaries, latency, transcript |
| `/sip-forward` | GET/POST | Plivo webhook — forwards to LiveKit SIP via `<User>` tag |
| `/hangup` | GET/POST | Call hangup handler |

## Commands

```bash
# Run both agent + server (default, for deployment)
python3.11 livekit_agent.py

# Run agent worker only (connects to LiveKit Cloud)
python3.11 livekit_agent.py dev

# Run FastAPI server only (health + SIP forwarding)
python3.11 livekit_agent.py serve

# Deploy to Railway
railway up

# View logs
railway logs
```

## Environment Variables

```
LIVEKIT_API_KEY=xxx          # From cloud.livekit.io
LIVEKIT_API_SECRET=xxx       # From cloud.livekit.io
LIVEKIT_URL=wss://xxx        # LiveKit Cloud WebSocket URL
LIVEKIT_SIP_URI=xxx          # SIP trunk host (e.g., trunk-id.sip.livekit.cloud)
OPENAI_API_KEY=xxx           # GPT-4o-mini LLM + call summary generation
DEEPGRAM_API_KEY=xxx         # Nova-2 STT + Aura TTS
PLIVO_AUTH_ID=xxx            # Telephony
PLIVO_AUTH_TOKEN=xxx
DATABASE_URL=xxx             # PostgreSQL (Railway provides this)
```

## LiveKit SIP Setup Checklist

1. **Inbound SIP Trunk** created in LiveKit Cloud dashboard
2. **Dispatch Rule** with agent name set to `"Livekit-demo"` (must match `agent_name` in code)
3. **Plivo Application** answer URL pointed to `<railway-url>/sip-forward` (POST)
4. **`LIVEKIT_SIP_URI`** env var set to the SIP trunk host (e.g., `trunk-id.sip.livekit.cloud`)

## Latency Optimizations

1. **Deepgram Nova-2 STT** (~100-200ms) instead of OpenAI Whisper (~500-1500ms)
2. **Deepgram Aura streaming TTS** (~200ms) instead of OpenAI TTS (~500ms+)
3. **Correct `<User>` SIP tag** for Plivo forwarding (was `<Number>` which doesn't work)
4. **Per-turn latency tracking** — measures time from `user_input_transcribed` to `agent_speech_committed`
5. **PostgreSQL call logging** persists across restarts (was in-memory)
6. **AI-initiated hangup** via `end_call` function tool with 3s TTS buffer

## Per-Turn Latency Tracking

The agent measures round-trip latency for each conversational turn:
- Timer starts on `user_input_transcribed` event (STT final transcript)
- Timer stops on `agent_speech_committed` event (first AI speech chunk)
- Per-turn values logged in Railway logs, average saved to `avg_latency_ms` in PostgreSQL
- Dual event handlers: `agent_speech_committed` (primary) + `conversation_item_added` (fallback with deduplication)

## AI Capabilities (Function Calling)

- `search_knowledge` — 20-topic keyword-scored knowledge base: company info, pricing, plans, hours, location, contact, features, security, free trial, return policy, shipping, warranty, password reset, account setup, technical issues, order status, bulk discounts, payment methods
- `lookup_order` — check order status by ID
- `schedule_callback` — schedule callback from sales, support, or billing
- `end_call` — AI-initiated hangup when caller says goodbye (3s delay for farewell TTS)

## Database Schema

Table: `call_logs`

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PRIMARY KEY | Auto-increment |
| call_id | VARCHAR(100) | LiveKit participant identity |
| start_time | TIMESTAMP | Call start |
| end_time | TIMESTAMP | Call end |
| duration_seconds | FLOAT | Call duration |
| summary | TEXT | AI-generated 1-2 sentence summary (GPT-4o-mini) |
| transcript | JSONB | Full conversation as `[["User", "..."], ["AI", "..."]]` |
| message_count | INTEGER | Total messages |
| avg_latency_ms | FLOAT | Average per-turn latency in milliseconds |
| created_at | TIMESTAMP | Record creation time |

Call data is saved synchronously in a `finally` block using the sync OpenAI client to guarantee persistence even during process shutdown.

## LiveKit vs Pipecat Comparison

| Aspect | LiveKit (this project) | Pipecat (`latency_improv/`) |
|--------|----------------------|---------------------------|
| Audio transport | SIP trunk (LiveKit Cloud handles routing) | WebSocket (raw mulaw 8kHz stream) |
| STT integration | LiveKit plugin — `deepgram.STT()` | Manual Deepgram WebSocket client |
| TTS integration | LiveKit plugin — `deepgram.TTS()` | Manual Deepgram REST + streaming chunks |
| Turn detection | LiveKit's built-in STT endpointing + Silero VAD | Manual silence detection (0.8s threshold) |
| Interruptions | Handled by LiveKit SDK automatically | Manual TTS cancel + audio buffer flush |
| Code complexity | ~610 lines (SDK handles audio pipeline) | ~1570 lines (manual audio pipeline) |
| Deployment | Two subprocesses (agent + FastAPI) | Single process (FastAPI + WebSocket) |
| Audio control | Abstracted by SDK | Full control over every audio frame |

**When to use LiveKit**: Simpler setup, less code to maintain, built-in turn detection and interruption handling. Better when you want to focus on AI logic rather than audio plumbing.

**When to use Pipecat**: Full control over audio pipeline, custom audio processing, fine-tuned silence detection thresholds, direct WebSocket streaming without SIP intermediary.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Telephony | Plivo |
| SIP | LiveKit Cloud SIP Trunk |
| STT | Deepgram Nova-2 |
| LLM | GPT-4o-mini |
| TTS | Deepgram Aura (asteria-en voice) |
| Agent SDK | LiveKit Agents v1.4 |
| Web Server | FastAPI + Uvicorn |
| Database | PostgreSQL (Railway) |
| Hosting | Railway |

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| 401 connecting to LiveKit | Wrong API key/secret | Verify `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in Railway env vars |
| Agent registers but never gets calls | Dispatch rule agent name mismatch | Set dispatch rule Agents field to `"Livekit-demo"` |
| Agent joins room but no audio | Invalid AgentSession params or wrong model names | Use `nova-2` (not nova-3), `aura-asteria-en` (not aura-2-asteria-en) |
| `UserInputTranscribedEvent` error | Event handler treats param as string | Access `event.transcript` attribute, not string directly |
| Call logs not saved | Async save doesn't complete before exit | Use sync `_save_call_sync()` in `finally` block |
| AI says "can't help" without trying | Missing tool call | Agent instructions say "ALWAYS try search_knowledge first" |
