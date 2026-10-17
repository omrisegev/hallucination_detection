| subset | n | P(chosen shorter) | med len ch/rej | tool-call ch/rej | prose ch/rej | mean_nll acc | mean_ent acc | len-matched n | nll acc (matched) | ent acc (matched) | nll acc (len-resid) | ent acc (len-resid) | sum_nll acc | same-format n | ent acc (same fmt) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GTA | 118 | 0.487 | 65/62 | 1.00/0.42 | 0.96/0.58 | 0.339 | 0.703 | 18 | 0.278 | 0.778 | 0.331 | 0.678 | 0.381 | 4 | 1.000 |
| ToolSandbox | 130 | 0.838 | 36/50 | 0.95/0.92 | 0.05/0.08 | 0.423 | 0.738 | 32 | 0.688 | 0.750 | 0.769 | 0.823 | 0.623 | 113 | 0.752 |
| bfcl | 111 | 0.514 | 28/27 | 1.00/1.00 | 0.00/0.00 | 0.441 | 0.486 | 64 | 0.391 | 0.438 | 0.414 | 0.477 | 0.423 | 111 | 0.486 |
| tooltalk | 86 | 0.634 | 66/69 | 1.00/1.00 | 0.00/0.00 | 0.581 | 0.314 | 48 | 0.667 | 0.292 | 0.581 | 0.326 | 0.767 | 86 | 0.314 |
| ALL | 445 | 0.625 | 43/52 | 0.98/0.82 | 0.27/0.18 | 0.436 | 0.584 | 162 | 0.519 | 0.494 | 0.528 | 0.602 | 0.537 | 314 | 0.541 |

GTA (chosen fmt, rejected fmt) counts: {('tool+prose', '+prose'): 68, ('tool+prose', 'tool'): 45, ('tool', 'tool'): 4, ('tool', '+prose'): 1}
