"""
title: Kika RAG Open WebUI Pipeline
author: Matthias Hellmann
version: 1.0
"""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from open_webui.models.users import Users, UserModel
from open_webui.utils.chat import generate_chat_completion

logger = logging.getLogger(__name__)


class Pipe:
    """
    A direct replica of Open WebUI's built-in chat pipeline that keeps web search
    and retrieval-augmented generation (RAG) behaviour intact. The pipeline simply
    forwards the incoming request back into the core chat completion flow so the
    standard preprocessing, model selection, web search, filters, and RAG steps
    all execute exactly as they would without the custom pipeline layer.
    """

    class Valves(BaseModel):
        model_id: Optional[str] = Field(
            default=None,
            description=(
                "Fallback downstream model id to use when the request's model field "
                "is the pipeline id (e.g. 'kikarag'). Provide any model id visible "
                "in the Models list such as 'openai/gpt-4o-mini'."
            ),
        )
        bypass_model_access_checks: bool = Field(
            default=False,
            description=(
                "When true, bypasses the built-in model access control checks so the "
                "pipeline can call models the user may not normally access."
            ),
        )
        # TODO: remove bypass_model_access_checks once underlying access rules are confirmed.
        pass

    class UserValves(BaseModel):
        model_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional per-user override of the downstream model id. If provided, "
                "this takes precedence over the pipeline-level valve."
            ),
        )
        pass

    def __init__(self) -> None:
        self.id = "kikarag"
        self.name = "Kika RAG Open WebUI Pipeline"
        self.description = (
            "Custom RAG enabled pipeline."
        )
        self.type = "pipe"
        self.valves = self.Valves()
        logger.debug("Pipeline %s initialised", self.id)

    def _resolve_user_model(
        self,
        user_payload: Optional[dict],
        metadata: Optional[Dict[str, Any]],
    ) -> UserModel:
        candidate_id = None
        if user_payload:
            candidate_id = (
                user_payload.get("id")
                or user_payload.get("user_id")
                or user_payload.get("sub")
            )

        if not candidate_id and metadata:
            candidate_id = metadata.get("user_id") or metadata.get("id")

        if not candidate_id:
            raise ValueError(
                "Unable to resolve user id from pipeline context; ensure user "
                "information is forwarded."
            )

        user = Users.get_user_by_id(candidate_id)
        if user:
            return user

        payload = dict(user_payload or {})
        payload.setdefault("id", candidate_id)
        payload.setdefault("name", payload.get("username", "Pipeline User"))
        payload.setdefault("email", payload.get("email") or f"{candidate_id}@local")
        payload.setdefault("role", payload.get("role", "user"))
        payload.setdefault("profile_image_url", payload.get("profile_image_url", "/user.png"))
        now_ts = int(time.time())
        payload.setdefault("last_active_at", now_ts)
        payload.setdefault("updated_at", now_ts)
        payload.setdefault("created_at", now_ts)

        return UserModel.model_validate(payload)

    def _resolve_target_model(
        self,
        incoming_model: Optional[str],
        metadata: Optional[Dict[str, Any]],
        user_payload: Optional[dict],
        model_info: Optional[Dict[str, Any]],
        request,
    ) -> str:
        if incoming_model:
            if incoming_model != self.id and "." not in incoming_model:
                return incoming_model

            if "." in incoming_model:
                pipeline_id, sub_model_id = incoming_model.split(".", 1)
                if pipeline_id == self.id and sub_model_id:
                    return sub_model_id
                if pipeline_id != self.id:
                    return incoming_model

        if model_info:
            info_blob = model_info.get("info") or {}
            candidate = info_blob.get("base_model_id")
            if isinstance(candidate, str) and candidate and candidate != self.id:
                return candidate

            params = info_blob.get("params") or {}
            candidate = params.get("model") or params.get("base_model_id")
            if isinstance(candidate, str) and candidate:
                return candidate

            meta = info_blob.get("meta") or {}
            candidate = (
                meta.get("model")
                or meta.get("base_model_id")
                or meta.get("target_model_id")
            )
            if isinstance(candidate, str) and candidate and candidate != self.id:
                return candidate

        if metadata:
            for key in ("model", "model_id", "target_model_id", "base_model_id"):
                value = metadata.get(key)
                if isinstance(value, str) and value:
                    return value

        if user_payload:
            user_valves = user_payload.get("valves")
            if isinstance(user_valves, dict):
                candidate = user_valves.get("model_id")
                if isinstance(candidate, str) and candidate:
                    return candidate

        if self.valves.model_id:
            return self.valves.model_id

        # Fallback to default models configuration if available
        if hasattr(request.app.state, "config"):
            default_models = getattr(request.app.state.config, "DEFAULT_MODELS", None)
            candidates = []
            if isinstance(default_models, str):
                try:
                    candidates = json.loads(default_models)
                except Exception:
                    if default_models:
                        candidates = [default_models]
            elif isinstance(default_models, (list, tuple)):
                candidates = list(default_models)

            for candidate in candidates:
                if (
                    isinstance(candidate, str)
                    and candidate
                    and candidate in request.app.state.MODELS
                    and request.app.state.MODELS[candidate].get("pipe") is None
                ):
                    return candidate

        # As a last resort, pick the first non-pipeline model
        for model_id, model in request.app.state.MODELS.items():
            if model_id == self.id:
                continue
            if model.get("pipe"):
                continue
            return model_id

        raise ValueError(
            "Unable to determine the downstream model id. Provide the request model "
            "as '<pipeline_id>.<model_id>' or configure a model id in the pipeline "
            "valves."
        )

    def _prepare_form_data(
        self, body: dict, metadata: Optional[Dict[str, Any]], target_model: str
    ) -> dict:
        form_data = copy.deepcopy(body)
        form_data.pop("user", None)
        form_data["model"] = target_model
        if metadata is not None:
            form_data["metadata"] = metadata
        return form_data

    async def pipe(
        self,
        body: dict,
        __request__=None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if __request__ is None:
            raise ValueError("FastAPI request context (__request__) is required.")

        user_model = self._resolve_user_model(__user__, __metadata__)
        target_model = self._resolve_target_model(
            body.get("model"),
            __metadata__,
            __user__,
            __model__,
            __request__,
        )
        form_data = self._prepare_form_data(body, __metadata__, target_model)

        logger.debug(
            "Dispatching pipeline request for user %s to model '%s'",
            user_model.id,
            target_model,
        )

        response = await generate_chat_completion(
            __request__,
            form_data,
            user=user_model,
            bypass_filter=self.valves.bypass_model_access_checks,
        )
        return response
