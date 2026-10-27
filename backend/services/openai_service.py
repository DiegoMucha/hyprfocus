import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config import Config
from system_prompt import HYPER_SYSTEM_PROMPT, build_hyper_prompt


logger = logging.getLogger(__name__)


class OpenAIService:
    """OpenAI Responses API adapter for Hyper's existing command contract."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        self.timeout = Config.OPENAI_TIMEOUT_SECONDS
        self.max_retries = Config.OPENAI_MAX_RETRIES

    def _call_openai_api(
        self, prompt: str, system_instruction: str = "", as_json: bool = True
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY no está configurada. Defínela en backend/.env."
            )

        request_input = prompt
        if as_json:
            request_input = (
                "Return the response as a valid JSON object.\n\n" + prompt
            )

        body: Dict[str, Any] = {
            "model": self.model,
            "input": request_input,
        }
        if system_instruction:
            body["instructions"] = system_instruction
        if as_json:
            body["text"] = {"format": {"type": "json_object"}}

        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._extract_output_text(payload)
            except urllib.error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP {exc.code}: {details}"
                retryable = exc.code in (408, 409, 429) or exc.code >= 500
                if not retryable or attempt >= self.max_retries:
                    break
                logger.warning("OpenAI temporalmente no disponible: %s", last_error)
            except (TimeoutError, urllib.error.URLError) as exc:
                last_error = str(exc)
                if attempt >= self.max_retries:
                    break
                logger.warning("Error transitorio conectando con OpenAI: %s", exc)

            time.sleep(2**attempt)

        raise RuntimeError(f"Fallo al invocar la API de OpenAI: {last_error}")

    @staticmethod
    def _extract_output_text(payload: Dict[str, Any]) -> str:
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    return str(content["text"]).strip()
        error = payload.get("error")
        raise RuntimeError(f"Respuesta inesperada de OpenAI: {error or payload}")

    def interpret_user_command(
        self,
        user_message: str,
        command_id: str,
        active_commands: Optional[List[Dict[str, Any]]] = None,
        music_catalog: Optional[List[str]] = None,
        pending_clarification: Optional[Dict[str, Any]] = None,
        document_context: Optional[str] = None,
        vision_event: Optional[str] = None,
    ) -> Dict[str, Any]:
        prompt = build_hyper_prompt(
            user_message=user_message,
            command_id=command_id,
            active_commands=active_commands,
            music_catalog=music_catalog,
            pending_clarification=pending_clarification,
            document_context=document_context,
            vision_event=vision_event,
        )
        raw_response = self._call_openai_api(
            prompt=prompt,
            system_instruction=HYPER_SYSTEM_PROMPT,
            as_json=True,
        )
        try:
            parsed = json.loads(raw_response)
            if "ID_comando" not in parsed or not parsed["ID_comando"]:
                parsed["ID_comando"] = command_id
            return parsed
        except json.JSONDecodeError as exc:
            logger.error("Error parseando JSON de OpenAI: %s -> %s", raw_response, exc)
            return {
                "ID_comando": command_id,
                "Comando": "NA",
                "Mensaje para responder al usuario": (
                    "No entendí el comando, por favor repite otra vez."
                ),
            }

    def generate_full_summary(
        self, content_text: str, user_instructions: Optional[str] = None
    ) -> str:
        prompt = (
            "Por favor elabora un resumen de estudio completo, estructurado y claro "
            "del siguiente material.\n"
            f"Instrucciones adicionales del estudiante: {user_instructions or 'Resumen organizado con puntos clave, conceptos esenciales y conclusiones'}.\n\n"
            f"--- CONTENIDO A RESUMIR ---\n{content_text[:20000]}\n"
        )
        return self._call_openai_api(
            prompt,
            system_instruction="Eres un tutor académico de alto nivel.",
            as_json=False,
        )

    def generate_flashcards_content(
        self, topic_or_text: str, count: int = 5
    ) -> List[Dict[str, str]]:
        prompt = (
            f"Genera exactamente {count} tarjetas de estudio sobre:\n"
            f"{topic_or_text[:15000]}\n\n"
            "Responde como un objeto JSON con la propiedad 'flashcards', que debe "
            "ser un arreglo de objetos con las llaves 'pregunta' y 'respuesta'."
        )
        response = self._call_openai_api(
            prompt,
            system_instruction=(
                "Eres un generador de flashcards educativas concisas y pedagógicas."
            ),
            as_json=True,
        )
        try:
            parsed = json.loads(response)
            cards = parsed.get("flashcards", []) if isinstance(parsed, dict) else []
            return cards if isinstance(cards, list) else []
        except Exception as exc:
            logger.error("Error parseando flashcards: %s", exc)
            return [
                {
                    "pregunta": f"Tema: {topic_or_text[:50]}",
                    "respuesta": "Repaso del tema solicitado.",
                }
            ]
