import json
import logging
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from config import Config
from system_prompt import HYPER_SYSTEM_PROMPT, build_hyper_prompt

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model = model or Config.GEMINI_MODEL
        # Modelos alternativos en caso de que 3.8 flash esté en preview o requiera fallback
        self.fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

    def _call_gemini_api(self, prompt: str, system_instruction: str = "", as_json: bool = True) -> str:
        """
        Llama a la API REST de Google Gemini sin dependencias pesadas.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY no está configurada. Por favor define tu API key en .env o config.py")

        models_to_try = [self.model] + [m for m in self.fallback_models if m != self.model]

        last_error = None
        for current_model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{current_model}:generateContent?key={self.api_key}"
            
            body = {
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1
                }
            }

            if system_instruction:
                body["systemInstruction"] = {
                    "parts": [{"text": system_instruction}]
                }

            if as_json:
                body["generationConfig"]["responseMimeType"] = "application/json"

            json_data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=json_data,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    res_body = response.read().decode("utf-8")
                    data = json.loads(res_body)
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
                    raise ValueError(f"Respuesta inesperada de Gemini: {data}")
            except urllib.error.HTTPError as e:
                err_content = e.read().decode("utf-8", errors="ignore")
                logger.warning(f"Error HTTP {e.code} con modelo {current_model}: {err_content}")
                last_error = f"HTTP {e.code}: {err_content}"
                if e.code == 404:
                    # Modelo no encontrado, intentar el siguiente
                    continue
                else:
                    break
            except Exception as e:
                logger.error(f"Error al conectar con Gemini: {e}")
                last_error = str(e)
                break

        raise RuntimeError(f"Fallo al invocar la API de Gemini: {last_error}")

    def interpret_user_command(
        self,
        user_message: str,
        command_id: str,
        active_commands: Optional[List[Dict[str, Any]]] = None,
        music_catalog: Optional[List[str]] = None,
        pending_clarification: Optional[Dict[str, Any]] = None,
        document_context: Optional[str] = None,
        vision_event: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Interpreta la intención del usuario y devuelve el objeto JSON exacto especificado
        en funcionalidades.md
        """
        prompt = build_hyper_prompt(
            user_message=user_message,
            command_id=command_id,
            active_commands=active_commands,
            music_catalog=music_catalog,
            pending_clarification=pending_clarification,
            document_context=document_context,
            vision_event=vision_event
        )

        raw_response = self._call_gemini_api(
            prompt=prompt,
            system_instruction=HYPER_SYSTEM_PROMPT,
            as_json=True
        )

        # Limpiar posibles bloques markdown si el modelo los incluyó
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            # Asegurar que ID_comando coincida con el generado
            if "ID_comando" not in parsed or not parsed["ID_comando"]:
                parsed["ID_comando"] = command_id
            return parsed
        except json.JSONDecodeError as err:
            logger.error(f"Error parseando JSON de Gemini: {raw_response} -> {err}")
            return {
                "ID_comando": command_id,
                "Comando": "NA",
                "Mensaje para responder al usuario": "No entendí el comando, por favor repite otra vez."
            }

    def generate_full_summary(self, content_text: str, user_instructions: Optional[str] = None) -> str:
        """
        Genera el resumen completo de un texto o documento cargado.
        """
        prompt = (
            f"Por favor elabora un resumen de estudio completo, estructurado y claro del siguiente material.\n"
            f"Instrucciones adicionales del estudiante: {user_instructions or 'Resumen organizado con puntos clave, conceptos esenciales y conclusiones'}.\n\n"
            f"--- CONTENIDO A RESUMIR ---\n{content_text[:20000]}\n"
        )
        return self._call_gemini_api(prompt, system_instruction="Eres un tutor académico de alto nivel.", as_json=False)

    def generate_flashcards_content(self, topic_or_text: str, count: int = 5) -> List[Dict[str, str]]:
        """
        Genera un listado de flashcards (Pregunta / Respuesta) en formato JSON.
        """
        prompt = (
            f"Genera exactamente {count} tarjetas de estudio (flashcards) sobre el siguiente tema o contenido:\n"
            f"{topic_or_text[:15000]}\n\n"
            f"Responde estrictamente con un arreglo JSON de objetos con las llaves 'pregunta' y 'respuesta'.\n"
            f'Ejemplo: [{{"pregunta": "¿Qué es la fotosíntesis?", "respuesta": "Proceso mediante el cual las plantas convierten luz en energía química."}}]'
        )
        res = self._call_gemini_api(
            prompt,
            system_instruction="Eres un generador de flashcards educativas concisas y pedagógicas.",
            as_json=True
        )
        try:
            cleaned = res.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cards = json.loads(cleaned)
            if isinstance(cards, dict) and "flashcards" in cards:
                cards = cards["flashcards"]
            return cards if isinstance(cards, list) else []
        except Exception as e:
            logger.error(f"Error parseando flashcards: {e}")
            return [
                {"pregunta": f"Tema: {topic_or_text[:50]}", "respuesta": "Repaso del tema solicitado."}
            ]
