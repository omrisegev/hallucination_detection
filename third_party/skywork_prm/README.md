Vendored from https://github.com/SkyworkAI/skywork-o1-prm-inference at
`719b56b17447405e0f10e6c0360a581cf4ffa9c1`, Apache-2.0 (LICENSE).

`model_utils/prm_model.py`, `model_utils/io_utils.py`, and the required transitive
loader `model_utils/modeling_base.py` are byte-identical upstream files. Empty
package initializers were added. No upstream function was edited.

The upstream helper returns `(input_ids, steps, reward_flags)` (its wrapper calls
that second result an attention mask but discards it). It appends one terminal
reward tag after the trailing newline. The adapter preserves all input IDs and
records that terminal position separately; policy is documented in the handoff.
