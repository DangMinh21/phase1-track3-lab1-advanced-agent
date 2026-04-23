from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, PLANNER_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


@dataclass
class RuntimeOutput:
    text: str
    total_tokens: int
    latency_ms: int


class OpenAICompatibleRuntime:
    def __init__(self, model: str, api_key: str, base_url: str | None = None, timeout: float = 60.0) -> None:
        self.model = model
        normalized_base_url = (base_url or "").strip() or None
        if normalized_base_url and not normalized_base_url.startswith(("http://", "https://")):
            raise ValueError(
                "base_url must include scheme, for example: https://api.openai.com/v1"
            )
        self.client = OpenAI(api_key=api_key, base_url=normalized_base_url, timeout=timeout)

    def _chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> RuntimeOutput:
        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        text = response.choices[0].message.content or ""
        usage = response.usage
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return RuntimeOutput(text=text.strip(), total_tokens=total_tokens, latency_ms=latency_ms)

    @staticmethod
    def _context_text(example: QAExample) -> str:
        blocks = []
        for idx, chunk in enumerate(example.context, start=1):
            blocks.append(f"[{idx}] {chunk.title}: {chunk.text}")
        return "\n".join(blocks)

    @staticmethod
    def _memory_text(reflection_memory: list[str], max_items: int = 3) -> str:
        if not reflection_memory:
            return "(empty)"
        trimmed = reflection_memory[-max_items:]
        return "\n".join(f"- {item}" for item in trimmed)

    @staticmethod
    def _parse_plan(text: str) -> list[str]:
        try:
            payload = json.loads(text)
            raw_plan = payload.get("plan", [])
            if isinstance(raw_plan, list):
                steps = [str(item).strip() for item in raw_plan if str(item).strip()]
                if steps:
                    return steps[:3]
        except Exception:
            pass
        return [
            "Find evidence for first-hop entity in context.",
            "Resolve second-hop entity from linked evidence.",
            "Return the final grounded answer string.",
        ]

    def _actor_generate(
        self,
        example: QAExample,
        attempt_id: int,
        reflection_memory: list[str],
        plan_steps: list[str],
        branch_hint: str,
    ) -> RuntimeOutput:
        plan_text = "\n".join(f"- {step}" for step in plan_steps)
        prompt = (
            f"Question:\n{example.question}\n\n"
            f"Context:\n{self._context_text(example)}\n\n"
            f"Attempt: {attempt_id}\n"
            f"Reflection memory:\n{self._memory_text(reflection_memory)}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Branch hint:\n{branch_hint}\n\n"
            "Return only the final answer string."
        )
        return self._chat(ACTOR_SYSTEM, prompt, json_mode=False)

    @staticmethod
    def _select_answer(candidates: list[str]) -> str:
        cleaned = [c.strip() for c in candidates if c and c.strip()]
        if not cleaned:
            return "insufficient evidence"
        counts: dict[str, int] = {}
        canonical_to_raw: dict[str, str] = {}
        for ans in cleaned:
            key = normalize_answer(ans)
            counts[key] = counts.get(key, 0) + 1
            canonical_to_raw.setdefault(key, ans)
        best_key = max(counts, key=counts.get)
        return canonical_to_raw[best_key]

    def actor_answer(self, example: QAExample, attempt_id: int, reflection_memory: list[str]) -> tuple[str, int, int]:
        # plan_then_execute: create an explicit short plan before answer generation
        planner_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Context:\n{self._context_text(example)}\n\n"
            f"Attempt: {attempt_id}\n"
            f"Reflection memory:\n{self._memory_text(reflection_memory)}\n\n"
            'Return JSON only with key "plan".'
        )
        plan_out = self._chat(PLANNER_SYSTEM, planner_prompt, json_mode=True)
        plan_steps = self._parse_plan(plan_out.text)

        primary = self._actor_generate(
            example=example,
            attempt_id=attempt_id,
            reflection_memory=reflection_memory,
            plan_steps=plan_steps,
            branch_hint="Primary reasoning path: prioritize direct evidence links between hop-1 and hop-2.",
        )
        candidates = [primary.text]
        total_tokens = plan_out.total_tokens + primary.total_tokens
        total_latency = plan_out.latency_ms + primary.latency_ms

        # mini_lats_branching: run one lightweight alternate branch when reflection exists
        if reflection_memory or attempt_id > 1:
            alt = self._actor_generate(
                example=example,
                attempt_id=attempt_id,
                reflection_memory=reflection_memory,
                plan_steps=plan_steps,
                branch_hint="Alternate path: challenge the current entity and verify a second candidate from context.",
            )
            candidates.append(alt.text)
            total_tokens += alt.total_tokens
            total_latency += alt.latency_ms

        # self_consistency_vote: choose the normalized majority answer across branches
        final_answer = self._select_answer(candidates)
        return final_answer, total_tokens, total_latency

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
        prompt = (
            f"Question:\n{example.question}\n\n"
            f"Gold answer:\n{example.gold_answer}\n\n"
            f"Predicted answer:\n{answer}\n\n"
            "Return JSON only with keys: score, reason, missing_evidence, spurious_claims"
        )

        last_text = ""
        total_tokens = 0
        total_latency = 0
        for _ in range(3):
            out = self._chat(EVALUATOR_SYSTEM, prompt, json_mode=True)
            total_tokens += out.total_tokens
            total_latency += out.latency_ms
            last_text = out.text
            try:
                parsed = json.loads(out.text)
                judge = JudgeResult.model_validate(parsed)
                return judge, total_tokens, total_latency
            except Exception:
                prompt += "\nYour previous response was invalid JSON. Return valid JSON only."

        fallback = JudgeResult(
            score=0,
            reason="Evaluator returned invalid JSON after retries.",
            missing_evidence=["Evaluator output invalid JSON."],
            spurious_claims=[last_text[:200]],
        )
        return fallback, total_tokens, total_latency

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
        prompt = (
            f"Question:\n{example.question}\n\n"
            f"Attempt id: {attempt_id}\n"
            f"Evaluator reason: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
            f"Spurious claims: {judge.spurious_claims}\n\n"
            "Return JSON only with keys: attempt_id, failure_reason, lesson, next_strategy"
        )

        out = self._chat(REFLECTOR_SYSTEM, prompt, json_mode=True)
        try:
            parsed = json.loads(out.text)
            reflection = ReflectionEntry.model_validate(parsed)
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Previous answer was not fully grounded in context.",
                next_strategy="Re-check both hops and verify final entity against the second context chunk.",
            )
        return reflection, out.total_tokens, out.latency_ms
