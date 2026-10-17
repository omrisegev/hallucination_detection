# Rendered fixtures (no real records)

## apb fixture 1
UID: `apb:bfcl:0`

```text
<|im_start|>system
system<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 0 START⟧abc⟦STEP 0 END⟧<|im_end|>
<|im_start|>user
<tool_response>
def
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 1 START⟧xyz⟦STEP 1 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      113,
      116
    ],
    [
      232,
      235
    ]
  ],
  "steps": [
    {
      "message_index": 2,
      "think_block_inserted": true,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        102,
        102
      ],
      "content_char_span": [
        113,
        116
      ],
      "expected_text": "abc"
    },
    {
      "message_index": 4,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        221,
        221
      ],
      "content_char_span": [
        232,
        235
      ],
      "expected_text": "xyz"
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        25
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        53,
        61
      ]
    },
    {
      "role": "tool",
      "message_index": 3,
      "char_span": [
        160,
        163
      ]
    }
  ]
}
```

## apb fixture 2
UID: `apb:bfcl:1`

```text
<|im_start|>system
system<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 0 START⟧abc⟦STEP 0 END⟧<|im_end|>
<|im_start|>user
<tool_response>
def
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 1 START⟧xyz⟦STEP 1 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      113,
      116
    ],
    [
      232,
      235
    ]
  ],
  "steps": [
    {
      "message_index": 2,
      "think_block_inserted": true,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        102,
        102
      ],
      "content_char_span": [
        113,
        116
      ],
      "expected_text": "abc"
    },
    {
      "message_index": 4,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        221,
        221
      ],
      "content_char_span": [
        232,
        235
      ],
      "expected_text": "xyz"
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        25
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        53,
        61
      ]
    },
    {
      "role": "tool",
      "message_index": 3,
      "char_span": [
        160,
        163
      ]
    }
  ]
}
```

## tprm fixture 1
UID: `tprm:tooltalk:fixture:A`

```text
<|im_start|>system
system<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 0 START⟧<tool_call>
{"name": "b", "arguments": {}}
</tool_call>⟦STEP 0 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      113,
      168
    ]
  ],
  "steps": [
    {
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        102,
        102
      ],
      "content_char_span": [
        113,
        168
      ],
      "expected_text": "<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>"
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        25
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        53,
        61
      ]
    }
  ]
}
```

## tprm fixture 2
UID: `tprm:tooltalk:fixture:B`

```text
<|im_start|>system
system<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

⟦STEP 0 START⟧<tool_call>
{"name": "a", "arguments": {}}
</tool_call>⟦STEP 0 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      113,
      168
    ]
  ],
  "steps": [
    {
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        102,
        102
      ],
      "content_char_span": [
        113,
        168
      ],
      "expected_text": "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call>"
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        25
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        53,
        61
      ]
    }
  ]
}
```

## aft fixture 1
UID: `aft:math:0`

```text
<|im_start|>system
You are reviewing the log of a multi-agent system that tried to solve the task below.<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

[0] user:
⟦STEP 0 START⟧[Content] abc⟦STEP 0 END⟧

[1] agent:
⟦STEP 1 START⟧[Thought] def
[Action] xyz⟦STEP 1 END⟧

[2] tool:⟦STEP 2 START⟧⟦STEP 2 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      202,
      215
    ],
    [
      228,
      254
    ],
    [
      265,
      265
    ]
  ],
  "steps": [
    {
      "entry_index": 0,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        202,
        202
      ],
      "content_char_span": [
        202,
        215
      ],
      "expected_text": "[Content] abc",
      "field_char_spans": {
        "content": [
          212,
          215
        ]
      }
    },
    {
      "entry_index": 1,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        228,
        228
      ],
      "content_char_span": [
        228,
        254
      ],
      "expected_text": "[Thought] def\n[Action] xyz",
      "field_char_spans": {
        "thought": [
          238,
          241
        ],
        "action": [
          251,
          254
        ]
      }
    },
    {
      "entry_index": 2,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        265,
        265
      ],
      "content_char_span": [
        265,
        265
      ],
      "expected_text": "",
      "field_char_spans": {}
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        104
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        132,
        140
      ]
    }
  ]
}
```

## aft fixture 2
UID: `aft:math:1`

```text
<|im_start|>system
You are reviewing the log of a multi-agent system that tried to solve the task below.<|im_end|>
<|im_start|>user
question<|im_end|>
<|im_start|>assistant
<think>

</think>

[0] user:
⟦STEP 0 START⟧[Content] abc⟦STEP 0 END⟧

[1] agent:
⟦STEP 1 START⟧[Thought] def
[Action] xyz⟦STEP 1 END⟧

[2] tool:⟦STEP 2 START⟧⟦STEP 2 END⟧<|im_end|>

```

```json
{
  "step_char_spans": [
    [
      202,
      215
    ],
    [
      228,
      254
    ],
    [
      265,
      265
    ]
  ],
  "steps": [
    {
      "entry_index": 0,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        202,
        202
      ],
      "content_char_span": [
        202,
        215
      ],
      "expected_text": "[Content] abc",
      "field_char_spans": {
        "content": [
          212,
          215
        ]
      }
    },
    {
      "entry_index": 1,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        228,
        228
      ],
      "content_char_span": [
        228,
        254
      ],
      "expected_text": "[Thought] def\n[Action] xyz",
      "field_char_spans": {
        "thought": [
          238,
          241
        ],
        "action": [
          251,
          254
        ]
      }
    },
    {
      "entry_index": 2,
      "message_index": 2,
      "think_block_inserted": false,
      "has_source_reasoning": false,
      "reasoning_char_span": [
        265,
        265
      ],
      "content_char_span": [
        265,
        265
      ],
      "expected_text": "",
      "field_char_spans": {}
    }
  ],
  "context_spans": [
    {
      "role": "system",
      "message_index": 0,
      "char_span": [
        19,
        104
      ]
    },
    {
      "role": "user",
      "message_index": 1,
      "char_span": [
        132,
        140
      ]
    }
  ]
}
```
