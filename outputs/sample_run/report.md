# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_120.json
- Mode: real
- Records: 240
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.425 | 0.55 | 0.125 |
| Avg attempts | 1 | 2.025 | 1.025 |
| Avg token estimate | 3269.6 | 8483.54 | 5213.94 |
| Avg latency (ms) | 3766.81 | 10338.12 | 6571.31 |

## Failure modes
```json
{
  "by_agent": {
    "react": {
      "wrong_final_answer": 69,
      "none": 51
    },
    "reflexion": {
      "wrong_final_answer": 54,
      "none": 66
    }
  },
  "by_outcome": {
    "incorrect": {
      "wrong_final_answer": 123
    },
    "correct": {
      "none": 117
    }
  },
  "overall": {
    "wrong_final_answer": 123,
    "none": 117
  }
}
```

## Extensions implemented
- benchmark_report_json
- reflection_memory
- adaptive_max_attempts
- memory_compression
- reflection_overfit_guard
- structured_evaluator
- plan_then_execute
- mini_lats_branching
- self_consistency_vote

## Discussion
Reflexion showed the strongest gains on multi-hop questions where first-pass answers stopped early or selected the wrong second-hop entity. Compared with ReAct, EM improved at the cost of extra attempts, which increased token usage and end-to-end latency. Using structured evaluator output made the loop more stable by avoiding invalid parse branches. Reflection memory helped when failures repeated the same pattern, while adaptive stopping reduced wasteful retries in drift-heavy cases.
