from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


def _compress_memory(memory: list[str], max_items: int = 3) -> list[str]:
    if len(memory) <= max_items:
        return memory
    return memory[-max_items:]


def _classify_failure_mode(judge_reason: str) -> str:
    text = judge_reason.lower()
    if "second hop" in text or "first hop" in text:
        return "incomplete_multi_hop"
    if "drift" in text:
        return "entity_drift"
    if "loop" in text:
        return "looping"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: object | None = None

    def _should_continue(self, attempt_id: int, last_failure_mode: str, recent_failures: list[str]) -> bool:
        if attempt_id >= self.max_attempts:
            return False
        if self.agent_type != "reflexion":
            return False
        # adaptive_max_attempts: nếu drift liên tiếp 2 lần thì dừng sớm
        if len(recent_failures) >= 2 and recent_failures[-1] == "entity_drift" and recent_failures[-2] == "entity_drift":
            return False
        return True

    @staticmethod
    def _is_reflection_overfit(reflection_memory: list[str], traces: list[AttemptTrace]) -> bool:
        # reflection_overfit_guard: stop when strategy and wrong answer are repeated with no progress
        if len(reflection_memory) < 2 or len(traces) < 2:
            return False
        last_strategy = normalize_answer(reflection_memory[-1])
        prev_strategy = normalize_answer(reflection_memory[-2])
        if last_strategy != prev_strategy:
            return False
        last_answer = normalize_answer(traces[-1].answer)
        prev_answer = normalize_answer(traces[-2].answer)
        if last_answer != prev_answer:
            return False
        return traces[-1].score == 0 and traces[-2].score == 0

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        recent_failures: list[str] = []

        for attempt_id in range(1, self.max_attempts + 1):
            if self.runtime is None:
                answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                judge = evaluator(example, answer)
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
            else:
                answer, actor_tokens, actor_latency = self.runtime.actor_answer(example, attempt_id, reflection_memory)
                judge, eval_tokens, eval_latency = self.runtime.evaluator(example, answer)
                token_estimate = actor_tokens + eval_tokens
                latency_ms = actor_latency + eval_latency

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            final_answer = answer
            final_score = judge.score

            if judge.score == 1:
                traces.append(trace)
                break

            failure_mode = FAILURE_MODE_BY_QID.get(example.qid, _classify_failure_mode(judge.reason))
            recent_failures.append(failure_mode)

            if self._should_continue(attempt_id, failure_mode, recent_failures):
                if self.runtime is None:
                    reflection = reflector(example, attempt_id, judge)
                    ref_tokens = 0
                    ref_latency = 0
                else:
                    reflection, ref_tokens, ref_latency = self.runtime.reflector(example, attempt_id, judge)

                reflections.append(reflection)
                reflection_memory.append(reflection.next_strategy)
                reflection_memory = _compress_memory(reflection_memory, max_items=3)  # memory_compression
                trace.reflection = reflection
                trace.token_estimate += ref_tokens
                trace.latency_ms += ref_latency

            traces.append(trace)
            if self._is_reflection_overfit(reflection_memory, traces):
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime: object | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: object | None = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime)
