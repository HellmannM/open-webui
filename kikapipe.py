"""
title: Kika RAG Open WebUI Pipeline
author: Matthias Hellmann
version: 3.0
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from open_webui.config import DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE
from open_webui.constants import TASKS
from open_webui.models.models import Models
from open_webui.models.users import Users, UserModel
from open_webui.models.functions import Functions
from open_webui.routers.pipelines import process_pipeline_inlet_filter
from open_webui.retrieval.utils import get_sources_from_items
from open_webui.utils import middleware as _ow_middleware
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.misc import add_or_update_user_message, get_last_user_message
from open_webui.utils.task import (
    get_task_model_id,
    query_generation_template,
    rag_template,
)

PIPELINE_ID = "kikapipe_base"

logger = logging.getLogger(__name__)


if not getattr(_ow_middleware, "_kika_patch_installed", False):
    _original_chat_completion_files_handler = (
        _ow_middleware.chat_completion_files_handler
    )

    async def _kika_chat_completion_files_handler(request, body, extra_params, user):
        try:
            metadata = body.get("metadata") if isinstance(body, dict) else None
            custom_rag = (metadata or {}).get("custom_rag") or {}
            model_id = body.get("model") if isinstance(body, dict) else None
            model_prefix = (
                model_id.split(".", 1)[0] if isinstance(model_id, str) else None
            )

            if custom_rag.get("handled") or model_prefix == PIPELINE_ID:
                return body, {"sources": []}
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Custom RAG skip guard failed; falling back to default handler: %s",
                exc,
                exc_info=True,
            )
        return await _original_chat_completion_files_handler(
            request, body, extra_params, user
        )

    _ow_middleware.chat_completion_files_handler = _kika_chat_completion_files_handler
    _ow_middleware._kika_patch_installed = True
    _ow_middleware._kika_original_chat_completion_files_handler = (
        _original_chat_completion_files_handler
    )


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


@dataclass
class DocumentBundle:
    key: str
    display_name: Optional[str]
    source_info: Dict[str, Any]
    chunks: List[RetrievedChunk]


@dataclass
class CustomRagResult:
    query: str
    sources: List[Dict[str, Any]]
    bundles: List[DocumentBundle]
    stats: Dict[str, Any]
    queries: List[str]
    debug_mode: str
    debug_summary: Optional[str]


class Pipe:
    """
    Kika pipeline replaces Open WebUI's stock RAG pass with a custom implementation
    that runs entirely inside the pipeline. Knowledge attachments and web search
    inputs are intercepted before the core middleware can issue vector queries so
    we can perform retrieval, filtering, and context injection exactly once.
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
        top_k: Optional[int] = Field(
            default=None,
            description=(
                "Override the number of neighbours to retrieve from the vector store. "
                "Defaults to the global Open WebUI TOP_K setting."
            ),
        )
        top_k_reranker: Optional[int] = Field(
            default=None,
            description=(
                "Override the number of candidates to send to the reranker. "
                "Defaults to the global Open WebUI TOP_K_RERANKER setting."
            ),
        )
        relevance_threshold: Optional[float] = Field(
            default=None,
            description=(
                "Minimum similarity score (0-1 range) required to keep a chunk when "
                "using hybrid search."
            ),
        )
        hybrid_search: Optional[bool] = Field(
            default=None,
            description=(
                "Force-enable or disable hybrid (vector + BM25) retrieval regardless "
                "of the instance-level configuration."
            ),
        )
        full_context: Optional[bool] = Field(
            default=None,
            description=(
                "When true, force full-context retrieval for any attached collection."
            ),
        )
        debug_mode: Optional[str] = Field(
            default="off",
            description=(
                "Diagnostic output. Accepts 'off', 'chat', or 'log'. "
                "When 'chat', the pipeline appends a debug summary to the response; "
                "when 'log', it writes the summary to the server logs."
            ),
        )

    class UserValves(BaseModel):
        model_id: Optional[str] = Field(
            default=None,
            description=(
                "Optional per-user override of the downstream model id. If provided, "
                "this takes precedence over the pipeline-level valve."
            ),
        )
        top_k: Optional[int] = None
        top_k_reranker: Optional[int] = None
        relevance_threshold: Optional[float] = None
        hybrid_search: Optional[bool] = None
        full_context: Optional[bool] = None
        debug_mode: Optional[str] = None

    def __init__(self) -> None:
        self.id = PIPELINE_ID
        self.name = "Kika RAG Open WebUI Pipeline"
        self.description = "Custom RAG enabled pipeline."
        self.type = "pipe"
        self.valves = self.Valves()
        logger.debug("Pipeline %s initialised", self.id)

    def _extract_user_valves(
        self, user_payload: Optional[dict]
    ) -> Optional["Pipe.UserValves"]:
        if not user_payload:
            return None
        raw = user_payload.get("valves")
        if not isinstance(raw, dict):
            return None
        try:
            return self.UserValves(**raw)
        except Exception as exc:
            logger.debug("Failed to parse user valves: %s", exc, exc_info=True)
            return None

    def _merge_valves(self, user_payload: Optional[dict]) -> Dict[str, Any]:
        base = self.valves.model_dump(exclude_unset=True)
        user_valves = self._extract_user_valves(user_payload)
        if user_valves:
            overrides = user_valves.model_dump(exclude_unset=True)
            for key, value in overrides.items():
                if value is not None:
                    base[key] = value
        return base

    def _extract_query_and_context(
        self, messages: Sequence[Dict[str, Any]]
    ) -> tuple[Optional[str], Optional[str]]:
        if not messages:
            return None, None

        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if isinstance(content, list):
                # OpenAI-compatible message format
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content = part.get("text")
                        break

            if not isinstance(content, str):
                continue

            context = None
            if "<context>" in content and "</context>" in content:
                context = (
                    content.split("<context>", 1)[1].split("</context>", 1)[0].strip()
                )

            if "<user_query>" in content and "</user_query>" in content:
                query = (
                    content.split("<user_query>", 1)[1]
                    .split("</user_query>", 1)[0]
                    .strip()
                )
            elif "</context>" in content:
                query = content.split("</context>", 1)[1].strip()
            else:
                query = content.strip()

            return query or None, context

        return None, None

    def _bundle_sources(
        self,
        raw_sources: Sequence[Dict[str, Any]],
    ) -> tuple[List[DocumentBundle], List[Dict[str, Any]], Dict[str, Any]]:
        if not raw_sources:
            return (
                [],
                [],
                {
                    "raw_documents": 0,
                    "raw_chunks": 0,
                    "context_documents": 0,
                    "context_chunks": 0,
                },
            )

        bundles: "OrderedDict[str, DocumentBundle]" = OrderedDict()
        raw_chunk_count = 0

        for source in raw_sources:
            documents = source.get("document") or []
            metadatas = source.get("metadata") or []
            distances = source.get("distances") or []

            if distances and isinstance(distances[0], list):
                distances = distances[0]
            if not isinstance(distances, list):
                distances = list(distances) if distances else []

            raw_chunk_count += len(documents)

            for idx, (document_text, metadata) in enumerate(zip(documents, metadatas)):
                metadata = metadata or {}

                doc_key = (
                    metadata.get("source")
                    or metadata.get("file_id")
                    or metadata.get("collection_name")
                    or (source.get("source") or {}).get("id")
                    or (source.get("source") or {}).get("name")
                    or metadata.get("name")
                    or f"doc-{idx}"
                )
                doc_key = str(doc_key)

                bundle = bundles.get(doc_key)
                if not bundle:
                    display_name = (
                        metadata.get("name")
                        or metadata.get("title")
                        or (source.get("source") or {}).get("name")
                        or doc_key
                    )
                    bundle = DocumentBundle(
                        key=doc_key,
                        display_name=display_name,
                        source_info=source.get("source") or {},
                        chunks=[],
                    )
                    bundles[doc_key] = bundle

                distance = distances[idx] if idx < len(distances) else None
                if isinstance(distance, list) and distance:
                    distance = distance[0]

                bundle.chunks.append(
                    RetrievedChunk(
                        text=document_text,
                        metadata=metadata,
                        distance=(
                            distance if isinstance(distance, (int, float)) else None
                        ),
                    )
                )

        ui_sources: List[Dict[str, Any]] = []
        for bundle in bundles.values():
            payload = {
                "source": bundle.source_info,
                "document": [chunk.text for chunk in bundle.chunks],
                "metadata": [chunk.metadata for chunk in bundle.chunks],
            }
            distances = [
                chunk.distance for chunk in bundle.chunks if chunk.distance is not None
            ]
            if distances:
                payload["distances"] = distances
            ui_sources.append(payload)

        stats = {
            "raw_documents": len(bundles),
            "raw_chunks": raw_chunk_count,
            "context_documents": len(bundles),
            "context_chunks": sum(len(bundle.chunks) for bundle in bundles.values()),
        }

        return list(bundles.values()), ui_sources, stats

    def _render_context(
        self, bundles: Sequence[DocumentBundle]
    ) -> tuple[str, Dict[str, str]]:
        context_parts: List[str] = []
        citation_map: Dict[str, str] = {}
        next_idx = 1

        for bundle in bundles:
            citation_id = citation_map.get(bundle.key)
            if citation_id is None:
                citation_id = str(next_idx)
                citation_map[bundle.key] = citation_id
                next_idx += 1

            name_fragment = (
                f' name="{bundle.display_name}"' if bundle.display_name else ""
            )

            summary = bundle.display_name or bundle.key
            summary = (summary or "").strip()
            if not summary and bundle.chunks:
                fallback = (
                    bundle.chunks[0].metadata.get("title")
                    or bundle.chunks[0].metadata.get("name")
                    or bundle.chunks[0].metadata.get("source")
                    or ""
                )
                summary = fallback.strip()

            context_parts.append(
                f'<source id="{citation_id}"{name_fragment}>{summary}</source>'
            )

        return "\n".join(context_parts).strip(), citation_map

    def _inject_context(
        self,
        messages: List[Dict[str, Any]],
        context: str,
        query: str,
        request,
    ) -> List[Dict[str, Any]]:
        template = rag_template(
            request.app.state.config.RAG_TEMPLATE,
            context,
            query,
        )
        updated_messages = copy.deepcopy(messages)
        return add_or_update_user_message(template, updated_messages, append=False)

    @staticmethod
    def _deduplicate_files(files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: Dict[str, Dict[str, Any]] = {}
        for item in files or []:
            try:
                key = json.dumps(item, sort_keys=True)
            except TypeError:
                # Non-serialisable objects fallback to repr
                key = repr(item)
            unique[key] = item
        return list(unique.values())

    @staticmethod
    def _convert_knowledge_items(items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []

        converted: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("collection_name"):
                converted.append(
                    {
                        "id": item.get("collection_name"),
                        "name": item.get("name"),
                        "legacy": True,
                    }
                )
            elif item.get("collection_names"):
                converted.append(
                    {
                        "name": item.get("name"),
                        "type": "collection",
                        "collection_names": item.get("collection_names"),
                        "legacy": True,
                    }
                )
            else:
                converted.append(item)
        return converted

    def _extract_model_knowledge(
        self, model_blob: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not isinstance(model_blob, dict):
            return []
        info = model_blob.get("info") or {}
        meta = info.get("meta") or {}
        knowledge = meta.get("knowledge")
        if knowledge is None and "knowledge" in model_blob:
            knowledge = model_blob.get("knowledge")
        return self._convert_knowledge_items(knowledge)

    def _load_model_knowledge(self, model_id: Optional[str]) -> List[Dict[str, Any]]:
        if not isinstance(model_id, str) or not model_id:
            return []
        try:
            model_record = Models.get_model_by_id(model_id)
        except Exception:
            return []
        if not model_record:
            return []

        # model_record.meta might be a dict or Pydantic model
        meta_obj = getattr(model_record, "meta", None)
        if hasattr(meta_obj, "model_dump"):
            meta_dict = meta_obj.model_dump()
        elif isinstance(meta_obj, dict):
            meta_dict = meta_obj
        else:
            meta_dict = {}

        knowledge = meta_dict.get("knowledge")
        return self._convert_knowledge_items(knowledge)

    def _load_function_knowledge(
        self, function_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not isinstance(function_id, str) or not function_id:
            return []
        try:
            function_record = Functions.get_function_by_id(function_id)
        except Exception:
            return []
        if not function_record:
            return []

        meta_obj = getattr(function_record, "meta", None)
        if hasattr(meta_obj, "model_dump"):
            meta_dict = meta_obj.model_dump()
        elif isinstance(meta_obj, dict):
            meta_dict = meta_obj
        else:
            meta_dict = {}

        knowledge = meta_dict.get("knowledge")
        return self._convert_knowledge_items(knowledge)

    def _gather_files(
        self,
        metadata: Dict[str, Any],
        form_data: Dict[str, Any],
        selected_model_id: Optional[str],
        model_entry: Optional[Dict[str, Any]],
        base_model_id: Optional[str],
        external_files: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        files: List[Dict[str, Any]] = []
        seen_collections: set[str] = set()
        seen_file_ids: set[str] = set()

        def extend_items(
            origin: str, candidate_files: Optional[Sequence[Dict[str, Any]]]
        ):
            if isinstance(candidate_files, list) and candidate_files:
                logger.debug(
                    "Custom RAG: adding %s file entries from %s",
                    len(candidate_files),
                    origin,
                )
                files.extend(candidate_files)

        extend_items("external", external_files)
        extend_items("metadata", metadata.get("files"))
        extend_items("form_data", form_data.get("files"))

        def extend_knowledge(items: List[Dict[str, Any]]) -> None:
            for entry in items:
                if entry.get("type") == "collection":
                    identifier = entry.get("id") or entry.get("collection_name")
                    if identifier and identifier in seen_collections:
                        continue
                    if identifier:
                        seen_collections.add(identifier)
                if entry.get("type") == "file":
                    identifier = entry.get("id")
                    if identifier and identifier in seen_file_ids:
                        continue
                    if identifier:
                        seen_file_ids.add(identifier)
                files.append(entry)

        if isinstance(model_entry, dict):
            extend_knowledge(self._extract_model_knowledge(model_entry))

        metadata_model = metadata.get("model")
        if isinstance(metadata_model, dict):
            extend_knowledge(self._extract_model_knowledge(metadata_model))
        elif isinstance(metadata_model, str):
            extend_knowledge(self._load_model_knowledge(metadata_model))

        function_ids: set[str] = set()
        if isinstance(selected_model_id, str) and selected_model_id:
            function_ids.add(selected_model_id.split(".", 1)[0])
        function_ids.add(self.id)

        for function_id in function_ids:
            extend_knowledge(self._load_function_knowledge(function_id))

        candidate_ids: set[str] = set()
        if isinstance(selected_model_id, str) and selected_model_id:
            candidate_ids.add(selected_model_id.split(".", 1)[0])
        if isinstance(base_model_id, str) and base_model_id:
            candidate_ids.add(base_model_id)
        if isinstance(model_entry, dict):
            base_id = model_entry.get("base_model_id")
            if isinstance(base_id, str) and base_id:
                candidate_ids.add(base_id)
        for key in ("model", "model_id", "base_model_id", "target_model_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                candidate_ids.add(value)

        for candidate_id in candidate_ids:
            extend_knowledge(self._load_model_knowledge(candidate_id))

        deduped = self._deduplicate_files(files)
        logger.debug(
            "Custom RAG: gathered %s unique file/collection entries (selected_model=%s, base_model=%s, external=%s)",
            len(deduped),
            selected_model_id,
            base_model_id,
            len(external_files or []),
        )
        return deduped, []

    def _normalise_debug_mode(self, settings: Dict[str, Any]) -> str:
        value = settings.get("debug_mode")
        if not isinstance(value, str):
            return "off"
        mode = value.strip().lower()
        if mode in {"chat", "log"}:
            return mode
        return "off"

    def _is_pipeline_model(
        self, models: Dict[str, Any], model_id: Optional[str]
    ) -> bool:
        if not model_id or model_id not in models:
            return False
        entry = models.get(model_id) or {}
        if model_id == self.id:
            return True
        if entry.get("pipe"):
            return True
        pipeline_blob = entry.get("pipeline") or {}
        return pipeline_blob.get("type") == "pipe"

    def _resolve_query_model_id(
        self,
        preferred_id: Optional[str],
        fallback_id: Optional[str],
        models: Dict[str, Any],
    ) -> Optional[str]:
        for candidate in (preferred_id, fallback_id):
            if (
                isinstance(candidate, str)
                and candidate in models
                and not self._is_pipeline_model(models, candidate)
            ):
                return candidate

        for model_id, entry in models.items():
            if not self._is_pipeline_model(models, model_id):
                return model_id

        return None

    @staticmethod
    def _bounded_neighbor_count(value: Optional[int], default: int) -> int:
        if isinstance(value, int) and value > 0:
            return min(value, default)
        return default

    def _format_debug_summary(
        self,
        query: str,
        queries: List[str],
        bundles: Sequence[DocumentBundle],
        stats: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append("Kika RAG Debug")
        lines.append(f"Primary query: {query}")
        if len(queries) > 1 or queries[0] != query:
            lines.append("Generated queries:")
            for idx, candidate in enumerate(queries, 1):
                lines.append(f"  {idx}. {candidate}")
        else:
            lines.append(f"Generated queries: {queries[0]}")

        raw_docs = stats.get("raw_documents", 0)
        raw_chunks = stats.get("raw_chunks", 0)
        ctx_docs = stats.get("context_documents", raw_docs)
        ctx_chunks = stats.get("context_chunks", raw_chunks)
        lines.append(f"Retrieved docs: {raw_docs} | chunks: {raw_chunks}")
        if raw_docs != ctx_docs or raw_chunks != ctx_chunks:
            lines.append(f"Context docs: {ctx_docs} | chunks: {ctx_chunks}")
        candidate_models = stats.get("candidate_model_ids") or []
        if candidate_models:
            lines.append("Candidate model IDs:")
            for cid in candidate_models:
                lines.append(f"  - {cid}")

        max_docs = 5
        max_chunks = 3
        if not bundles:
            lines.append("No documents were retrieved for this request.")
        for doc_idx, bundle in enumerate(bundles[:max_docs], 1):
            name = bundle.display_name or bundle.key
            lines.append(f"{doc_idx}. {name}")
            for chunk_idx, chunk in enumerate(bundle.chunks[:max_chunks], 1):
                label = (
                    chunk.metadata.get("title")
                    or chunk.metadata.get("name")
                    or chunk.metadata.get("source")
                    or chunk.metadata.get("category")
                    or chunk.text
                )
                preview = (label or "").strip().replace("\n", " ")
                if len(preview) > 160:
                    preview = preview[:157] + "..."
                lines.append(f"    - Chunk {chunk_idx}: {preview}")
            omitted = max(0, len(bundle.chunks) - max_chunks)
            if omitted:
                lines.append(f"    - ... ({omitted} more chunks)")
        omitted_docs = max(0, len(bundles) - max_docs)
        if omitted_docs:
            lines.append(f"... ({omitted_docs} more documents)")

        return "\n".join(lines)

    async def _generate_queries(
        self,
        request,
        form_data: Dict[str, Any],
        default_query: str,
        user_model: UserModel,
    ) -> List[str]:
        config = getattr(request.app.state, "config", None)
        if not config or not getattr(config, "ENABLE_RETRIEVAL_QUERY_GENERATION", True):
            return [default_query]

        base_model_id = form_data.get("model")
        models = getattr(request.app.state, "MODELS", {}) or {}
        if not isinstance(base_model_id, str) or base_model_id not in models:
            return [default_query]

        preferred_task_model = get_task_model_id(
            base_model_id,
            getattr(config, "TASK_MODEL", None),
            getattr(config, "TASK_MODEL_EXTERNAL", None),
            models,
        )

        task_model_id = self._resolve_query_model_id(
            preferred_task_model,
            base_model_id,
            models,
        )
        if not task_model_id:
            return [default_query]

        template = getattr(config, "QUERY_GENERATION_PROMPT_TEMPLATE", "")
        if not isinstance(template, str) or not template.strip():
            template = DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE

        content = query_generation_template(
            template,
            form_data.get("messages", []),
            user_model,
        )

        metadata_snapshot = (
            request.state.metadata if hasattr(request.state, "metadata") else {}
        )
        payload = {
            "model": task_model_id,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "metadata": {
                **metadata_snapshot,
                "task": str(TASKS.QUERY_GENERATION),
                "task_body": form_data,
                "chat_id": form_data.get("chat_id", None),
            },
        }

        try:
            payload = await process_pipeline_inlet_filter(
                request,
                payload,
                user_model,
                models,
            )
        except Exception as exc:
            logger.debug("Query generation inlet filter failed: %s", exc)

        try:
            response = await generate_chat_completion(
                request,
                form_data=payload,
                user=user_model,
                bypass_filter=True,
            )
        except Exception as exc:
            logger.debug("Query generation failed; using prompt directly: %s", exc)
            return [default_query]

        if isinstance(response, JSONResponse):
            try:
                data = json.loads(response.body.decode("utf-8", "replace"))
            except Exception:
                data = None
        elif isinstance(response, dict):
            data = response
        else:
            data = None

        if not isinstance(data, dict):
            return [default_query]

        choices = data.get("choices") or []
        if not choices:
            return [default_query]

        message = choices[0].get("message") or {}
        content = (
            message.get("content") or message.get("reasoning_content") or ""
        ).strip()
        if not content:
            return [default_query]

        candidates: List[str] = []
        bracket_start = content.find("{")
        bracket_end = content.rfind("}") + 1 if "}" in content else 0
        if bracket_start != -1 and bracket_end > bracket_start:
            try:
                parsed = json.loads(content[bracket_start:bracket_end])
                queries = parsed.get("queries", [])
                if isinstance(queries, list):
                    candidates = [
                        str(item).strip()
                        for item in queries
                        if isinstance(item, str) and item.strip()
                    ]
            except Exception:
                candidates = []

        if not candidates:
            candidates = [content]

        return candidates or [default_query]

    async def _apply_custom_rag(
        self,
        form_data: Dict[str, Any],
        request,
        user_model: UserModel,
        user_payload: Optional[dict],
        selected_model_id: Optional[str],
        model_entry: Optional[Dict[str, Any]],
        base_model_id: Optional[str],
        external_files: Optional[List[Dict[str, Any]]],
    ) -> Optional[CustomRagResult]:
        metadata = copy.deepcopy(form_data.get("metadata") or {})
        existing_sources = copy.deepcopy(metadata.get("sources") or [])
        logger.debug(
            "Custom RAG: starting gather (selected_model=%s, base_model=%s, external_files=%s)",
            selected_model_id,
            base_model_id,
            len(external_files or []),
        )
        files, candidate_model_ids = self._gather_files(
            metadata,
            form_data,
            selected_model_id,
            model_entry,
            base_model_id,
            external_files,
        )
        logger.debug(
            "Custom RAG: files after gather=%s existing_sources=%s",
            len(files),
            len(existing_sources),
        )

        try:
            settings = self._merge_valves(user_payload)
        except Exception as exc:
            logger.debug(
                "Failed to merge valves for custom RAG: %s", exc, exc_info=True
            )
            settings = self.valves.model_dump(exclude_unset=True)

        debug_mode = self._normalise_debug_mode(settings)

        query, _ = self._extract_query_and_context(form_data.get("messages", []))
        if not query:
            query = get_last_user_message(form_data.get("messages", [])) or ""

        if not files and not existing_sources:
            if debug_mode == "off":
                return None

            queries = [query] if query else ["<empty query>"]
            stats = {
                "raw_documents": 0,
                "raw_chunks": 0,
                "context_documents": 0,
                "context_chunks": 0,
                "candidate_model_ids": candidate_model_ids,
            }

            metadata_copy = copy.deepcopy(metadata)
            metadata_copy.setdefault("custom_rag", {})
            metadata_copy["custom_rag"].update(
                {
                    "query": query,
                    "queries": queries,
                    "stats": stats,
                    "filters": stats,
                    "citation_map": {},
                    "mode": "skipped",
                    "candidate_model_ids": candidate_model_ids,
                    "handled": True,
                }
            )
            metadata_copy["files"] = files
            metadata_copy["sources"] = existing_sources

            debug_summary = self._format_debug_summary(
                query or "<empty query>",
                queries,
                [],
                stats,
            )
            if debug_mode == "log":
                logger.info(debug_summary)
            metadata_copy["custom_rag"]["debug_summary"] = debug_summary
            form_data["metadata"] = metadata_copy
            form_data.pop("files", None)

            return None

        if not hasattr(request.app.state, "EMBEDDING_FUNCTION"):
            logger.debug("Custom RAG: embedding function unavailable, skipping.")
            return None

        if request.app.state.EMBEDDING_FUNCTION is None:
            logger.debug("Custom RAG: embedding function is None, skipping.")
            return None

        config = request.app.state.config
        default_settings = {
            "top_k": getattr(config, "TOP_K", 4),
            "top_k_reranker": getattr(config, "TOP_K_RERANKER", 4),
            "relevance_threshold": getattr(config, "RELEVANCE_THRESHOLD", 0.0),
            "hybrid_search": getattr(config, "ENABLE_RAG_HYBRID_SEARCH", False),
            "full_context": getattr(config, "RAG_FULL_CONTEXT", False),
        }

        top_k = self._bounded_neighbor_count(
            settings.get("top_k"), default_settings["top_k"]
        )
        top_k_reranker = self._bounded_neighbor_count(
            settings.get("top_k_reranker"),
            default_settings["top_k_reranker"],
        )
        relevance_threshold = (
            settings.get("relevance_threshold")
            if settings.get("relevance_threshold") is not None
            else default_settings["relevance_threshold"]
        )
        hybrid_search = (
            settings.get("hybrid_search")
            if settings.get("hybrid_search") is not None
            else default_settings["hybrid_search"]
        )
        full_context_pref = (
            settings.get("full_context")
            if settings.get("full_context") is not None
            else default_settings["full_context"]
        )
        hybrid_search = bool(hybrid_search)
        full_context_pref = bool(full_context_pref)

        all_full_context = bool(files) and all(
            isinstance(item, dict) and item.get("context") == "full" for item in files
        )
        full_context = all_full_context or full_context_pref

        queries: List[str] = []
        if files and not all_full_context:
            queries = await self._generate_queries(
                request,
                form_data,
                query,
                user_model,
            )
        if not queries:
            fallback = (
                get_last_user_message(form_data.get("messages", [])) or query or ""
            )
            queries = [fallback]

        retrieval_mode = "supplied"
        sources = existing_sources

        if files:
            embedding_function = request.app.state.EMBEDDING_FUNCTION
            reranking_function = request.app.state.RERANKING_FUNCTION

            try:
                sources = get_sources_from_items(
                    request=request,
                    items=files,
                    queries=queries,
                    embedding_function=lambda texts, prefix=None: embedding_function(
                        texts,
                        prefix=prefix,
                        user=user_model,
                    ),
                    k=top_k,
                    reranking_function=(
                        (
                            lambda sentences: reranking_function(
                                sentences,
                                user=user_model,
                            )
                        )
                        if reranking_function
                        else None
                    ),
                    k_reranker=top_k_reranker,
                    r=relevance_threshold,
                    hybrid_bm25_weight=getattr(config, "HYBRID_BM25_WEIGHT", 0.5),
                    hybrid_search=hybrid_search,
                    full_context=full_context,
                    user=user_model,
                )
                retrieval_mode = "retrieved"
            except Exception as exc:
                logger.exception("Custom RAG retrieval failed: %s", exc)
                return None

        bundles, ui_sources, stats = self._bundle_sources(sources)
        stats["candidate_model_ids"] = candidate_model_ids
        stats["mode"] = retrieval_mode

        context_str, citation_map = self._render_context(bundles)
        if context_str:
            form_data["messages"] = self._inject_context(
                form_data.get("messages", []),
                context_str,
                query,
                request,
            )

        metadata_copy = copy.deepcopy(metadata)
        metadata_copy.setdefault("custom_rag", {})
        metadata_copy["custom_rag"].update(
            {
                "query": query,
                "queries": queries,
                "stats": stats,
                "filters": stats,
                "citation_map": citation_map,
                "mode": retrieval_mode,
                "candidate_model_ids": candidate_model_ids,
                "handled": True,
            }
        )
        metadata_copy["files"] = files
        metadata_copy["sources"] = ui_sources
        form_data["metadata"] = metadata_copy
        form_data.pop("files", None)

        debug_mode = self._normalise_debug_mode(settings)
        debug_summary = None
        if debug_mode != "off":
            debug_summary = self._format_debug_summary(
                query,
                queries,
                bundles,
                stats,
            )
            if debug_mode == "log" and debug_summary:
                logger.info(debug_summary)
            metadata_copy["custom_rag"]["debug_summary"] = debug_summary

        return CustomRagResult(
            query=query,
            sources=ui_sources,
            bundles=bundles,
            stats=stats,
            queries=queries,
            debug_mode=debug_mode,
            debug_summary=debug_summary,
        )

    async def _emit_sources_event(
        self,
        event_emitter,
        result: CustomRagResult,
    ) -> None:
        if not event_emitter or not result or not result.sources:
            return
        try:
            await event_emitter(
                {
                    "type": "chat:completion",
                    "data": {"sources": result.sources},
                }
            )
        except Exception as exc:
            logger.debug("Failed to emit custom RAG sources event: %s", exc)

    def _attach_sources_to_response(self, response, result: Optional[CustomRagResult]):
        if not result or not result.sources:
            return response

        if isinstance(response, StreamingResponse):
            original_response = response

            async def stream_wrapper():
                initial_payload = json.dumps({"sources": result.sources})
                yield f"data: {initial_payload}\n\n".encode("utf-8")
                async for chunk in original_response.body_iterator:
                    yield chunk

            return StreamingResponse(
                stream_wrapper(),
                media_type=original_response.media_type,
                headers=dict(original_response.headers),
                background=original_response.background,
            )

        if isinstance(response, JSONResponse):
            try:
                body = json.loads(response.body.decode("utf-8", "replace"))
            except Exception:
                body = {}
            body["sources"] = result.sources
            return JSONResponse(
                content=body,
                status_code=response.status_code,
                headers=response.headers,
            )

        if isinstance(response, dict):
            response = {**response, "sources": result.sources}
        return response

    def _attach_debug_to_response(
        self,
        response,
        result: Optional[CustomRagResult],
    ):
        if not result or result.debug_mode != "chat" or not result.debug_summary:
            return response

        summary = (
            f"\n\n---\n{result.debug_summary}\n\n" if result.debug_summary else ""
        )

        if isinstance(response, StreamingResponse):
            original_response = response

            async def stream_wrapper():
                if summary:
                    payload = json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "content": summary,
                                    },
                                    "index": 0,
                                }
                            ],
                            "debug": result.debug_summary,
                        }
                    )
                    yield f"data: {payload}\n\n".encode("utf-8")
                async for chunk in original_response.body_iterator:
                    yield chunk

            return StreamingResponse(
                stream_wrapper(),
                media_type=original_response.media_type,
                headers=dict(original_response.headers),
                background=original_response.background,
            )

        if isinstance(response, JSONResponse):
            try:
                data = json.loads(response.body.decode("utf-8", "replace"))
            except Exception:
                data = {}
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                    message["content"] = (
                        f"{content}{summary}"
                        if content and summary
                        else content or summary
                    )
                    data["choices"][0]["message"] = message
                    return JSONResponse(
                        content=data,
                        status_code=response.status_code,
                        headers=response.headers,
                    )
            return response

        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = message.get("content", "")
                    message["content"] = (
                        f"{content}{summary}"
                        if content and summary
                        else content or summary
                    )
            return response

        return response

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
        payload.setdefault(
            "profile_image_url", payload.get("profile_image_url", "/user.png")
        )
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

        user_valves = self._extract_user_valves(user_payload)
        if (
            user_valves
            and isinstance(user_valves.model_id, str)
            and user_valves.model_id
        ):
            return user_valves.model_id

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
        body_metadata = copy.deepcopy(body.get("metadata") or {})
        merged_metadata = copy.deepcopy(metadata) if metadata else {}

        for key, value in body_metadata.items():
            if key == "files" and value:
                merged_list = []
                merged_list.extend(merged_metadata.get("files") or [])
                merged_list.extend(value if isinstance(value, list) else [])
                if merged_list:
                    merged_metadata["files"] = merged_list
            elif key == "sources" and value:
                merged_sources = []
                merged_sources.extend(merged_metadata.get("sources") or [])
                merged_sources.extend(value if isinstance(value, list) else [])
                merged_metadata["sources"] = merged_sources
            else:
                merged_metadata.setdefault(key, value)

        form_data["metadata"] = merged_metadata
        return form_data

    async def pipe(
        self,
        body: dict,
        __request__=None,
        __user__: Optional[dict] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __model__: Optional[Dict[str, Any]] = None,
        __event_emitter__=None,
        __files__=None,
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
        metadata_for_debug = form_data.get("metadata") or {}
        logger.debug(
            "Custom RAG: pipe entry metadata files=%s sources=%s external_files=%s",
            len(metadata_for_debug.get("files") or []),
            len(metadata_for_debug.get("sources") or []),
            len(__files__ or []),
        )

        metadata_blob = form_data.get("metadata", {}) or {}
        metadata_model = metadata_blob.get("model")

        pipeline_model_id = body.get("model")
        selected_model_id = pipeline_model_id
        model_entry: Dict[str, Any] = {}
        if isinstance(pipeline_model_id, str):
            lookup_id = pipeline_model_id.split(".", 1)[0]
            model_entry = __request__.app.state.MODELS.get(lookup_id, {})

        if isinstance(metadata_model, dict) and metadata_model:
            model_entry = metadata_model

        custom_rag_result = None
        try:
            custom_rag_result = await self._apply_custom_rag(
                form_data,
                __request__,
                user_model,
                __user__,
                selected_model_id,
                model_entry,
                target_model,
                __files__,
            )
        except Exception as exc:
            logger.exception("Custom RAG pipeline failed: %s", exc)
            custom_rag_result = None

        if custom_rag_result:
            await self._emit_sources_event(__event_emitter__, custom_rag_result)

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
        response = self._attach_sources_to_response(response, custom_rag_result)
        response = self._attach_debug_to_response(response, custom_rag_result)
        return response
