# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.5 | 1.0 | 0.5 |
| Avg attempts | 1 | 1.5 | 0.5 |
| Avg token estimate | 385 | 790 | 405 |
| Avg latency (ms) | 200 | 455 | 255 |

## Failure modes
```json
{
  "react": {
    "none": 4,
    "incomplete_multi_hop": 1,
    "wrong_final_answer": 1,
    "entity_drift": 2
  },
  "reflexion": {
    "none": 8
  }
}
```

## Extensions implemented
- benchmark_report_json
- reflection_memory
- adaptive_max_attempts
- memory_compression
- mock_mode_for_autograding

## Discussion
Reflexion showed the strongest gains on multi-hop questions where first-pass answers stopped early or selected the wrong second-hop entity. Compared with ReAct, EM improved at the cost of extra attempts, which increased token usage and end-to-end latency. Using structured evaluator output made the loop more stable by avoiding invalid parse branches. Reflection memory helped when failures repeated the same pattern, while adaptive stopping reduced wasteful retries in drift-heavy cases.
