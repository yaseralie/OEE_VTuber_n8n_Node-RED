# src/open_llm_vtuber/conversations/single_conversation.py
from typing import Union, List, Dict, Any, Optional
import asyncio
import json
from loguru import logger
import numpy as np
import requests

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext

# Import output types from agent package (you said files tetap di agent/)
from ..agent.output_types import SentenceOutput, DisplayText, Actions


N8N_WEBHOOK_URL = "http://103.171.85.170/webhook/vtuber"


async def _post_to_n8n(payload: dict, timeout: int = 15) -> requests.Response:
    """
    Run a blocking requests.post in a thread so asyncio loop tidak terblokir.
    Mengembalikan objek requests.Response atau raise exception.
    """
    def _sync_post():
        return requests.post(N8N_WEBHOOK_URL, json=payload, timeout=timeout)

    return await asyncio.to_thread(_sync_post)


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Process a single-user conversation turn:
      - ambil input (text / asr)
      - kirim ke n8n webhook
      - bungkus respons jadi SentenceOutput (DisplayText + tts_text + Actions)
      - teruskan ke process_agent_output supaya UI + TTS + Live2D jalan
    """
    tts_manager = TTSTaskManager()
    full_response = ""

    try:
        # --- 1) kirim sinyal awal ke frontend ---
        await send_conversation_start_signals(websocket_send)
        logger.info(f"New Conversation Chain {session_emoji} started!")

        # --- 2) proses input user (ASR jika audio) ---
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send
        )

        # --- 3) buat batch input (dipakai oleh history / compatibility) ---
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=context.character_config.human_name,
            metadata=metadata,
        )

        # --- 4) simpan user message ke history (kecuali skip_history) ---
        skip_history = metadata and metadata.get("skip_history", False)
        if context.history_uid and not skip_history:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=context.character_config.human_name,
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        # --- 5) kirim ke n8n webhook (via requests dijalankan di thread) ---
        try:
            logger.info("Mengirim request ke n8n webhook...")
            resp = await _post_to_n8n({"text": input_text})
            reply_text = ""
            if resp.status_code == 200:
                # coba parse JSON, ambil field 'reply' bila ada
                try:
                    data = resp.json()
                    if isinstance(data, dict) and "reply" in data:
                        reply_text = data.get("reply", "")
                    else:
                        # jika JSON tapi beda struktur, gunakan string representation
                        reply_text = json.dumps(data) if not isinstance(data, str) else data
                except Exception:
                    # bukan JSON, fallback ke raw text
                    reply_text = resp.text or ""
            else:
                logger.warning(f"n8n balas status code {resp.status_code}")
                reply_text = f"[n8n error {resp.status_code}]"
            logger.info(f"Response dari n8n: {reply_text!r}")
        except Exception as e:
            logger.error(f"Gagal kirim ke n8n: {e}")
            reply_text = f"[Gagal konek ke n8n: {e}]"

        # --- 6) siapkan reply agar kompatibel dengan SentenceOutput dataclass ---
        # display_text harus berjenis DisplayText
        clean_reply = (reply_text or "").strip() or "Maaf, saya tidak dapat menjawab sekarang."
        display = DisplayText(text=clean_reply, name=context.character_config.character_name, avatar=context.character_config.avatar)
        actions = Actions()  # kosongkan actions; n8n bisa mengirim ekspresi nanti yg kita parse jika perlu

        sentence_output = SentenceOutput(display_text=display, tts_text=clean_reply, actions=actions)

        # --- 7) proses output melalui pipeline (akan meng-handle TTS/websocket/live2d) ---
        try:
            response_part = await process_agent_output(
                output=sentence_output,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
                translate_engine=getattr(context, "translate_engine", None),
            )
            response_part_str = str(response_part) if response_part is not None else ""
            full_response += response_part_str
        except Exception as e:
            logger.exception(f"Error while processing agent output pipeline: {e}")
            # fallback: kirim teks raw ke frontend
            try:
                await websocket_send(json.dumps({"type": "text", "text": clean_reply}))
            except Exception:
                pass
            full_response += clean_reply

        # --- 8) tunggu semua TTS task selesai ---
        if tts_manager.task_list:
            await asyncio.gather(*tts_manager.task_list, return_exceptions=True)
            try:
                await websocket_send(json.dumps({"type": "backend-synth-complete"}))
            except Exception:
                pass

        # --- 9) finalize conversation turn ---
        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
        )

        # --- 10) simpan ai response ke history jika ada ---
        if context.history_uid and full_response:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")

        return full_response

    except asyncio.CancelledError:
        logger.info(f"Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.exception(f"Unhandled error in conversation chain: {e}")
        try:
            await websocket_send(json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"}))
        except Exception:
            pass
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
