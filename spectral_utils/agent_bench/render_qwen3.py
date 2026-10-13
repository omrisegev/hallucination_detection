"""Native Qwen3 rendering with explicit scored-turn think conditioning."""
from copy import deepcopy
from bisect import bisect_left, bisect_right
import hashlib
import re
from .contracts import validate

HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>\n"
EMPTY = "<think>\n\n</think>\n\n"


def native(tokenizer, ms, tools, generation=False):
    return tokenizer.apply_chat_template(ms, tools=tools, tokenize=False,
        add_generation_prompt=generation, enable_thinking=False)


def render(item, tokenizer):
    row = validate(item)
    ms, tools = deepcopy(row["messages"]), row["tools"]
    for m in ms:
        if m.get("think_literal"):
            # Internal template control: preserve source tags as literal content.
            # No source reasoning_content field is accepted by the adapters.
            m["reasoning_content"] = ""
        if m["role"] not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"{row['uid']}: unsupported template role")
        # Literal delimiters would make parsing ambiguous. Report instead of modifying text.
        if "<|im_start|>" in m["content"] or "<|im_end|>" in m["content"]:
            raise ValueError(f"{row['uid']}: literal chat delimiter in source content")
    base = native(tokenizer, ms, tools)
    # Instrument context content only. Removing markers must prove native byte parity.
    probe = deepcopy(ms)
    markers = {}
    for i, m in enumerate(probe):
        if m["role"] != "assistant":
            tag = hashlib.sha256((row["uid"] + str(i)).encode()).hexdigest()
            left, right = f"@@AGENT_{tag}_BEGIN@@", f"@@AGENT_{tag}_END@@"
            if left in base or right in base:
                raise ValueError("context marker collision")
            m["content"] = left + m["content"] + right
            markers[i] = (left, right)
    marked = native(tokenizer, probe, tools)
    context = []
    events = sorted((marked.index(tag), i, kind, tag) for i, pair in markers.items()
                    for kind, tag in enumerate(pair))
    clean, cursor, starts = "", 0, {}
    for position, i, kind, tag in events:
        clean += marked[cursor:position]
        if kind == 0:
            starts[i] = len(clean)
        else:
            context.append({"role": ms[i]["role"], "message_index": i, "char_span": [starts[i], len(clean)]})
        cursor = position + len(tag)
    clean += marked[cursor:]
    if clean != base:
        raise ValueError(f"{row['uid']}: context locator changes template bytes")
    positions = [m.start() for m in re.finditer(re.escape(HEADER), base)]
    assistants = [i for i, m in enumerate(ms) if m["role"] == "assistant"]
    if len(positions) != len(assistants):
        raise ValueError("assistant template header count mismatch")
    scored = row["step_message_indices"] if row["render_mode"] == "chat" else [2]
    replacements = []
    for i, left in zip(assistants, positions):
        if i not in scored:
            continue
        start = left + len(HEADER)
        stop = base.index(END, start)
        body = base[start:stop]
        inserted = not body.startswith("<think>\n")
        source_reasoning = "source_reasoning" in ms[i]
        reasoning = ms[i].get("source_reasoning", "").strip("\n")
        if source_reasoning:
            # Render the tool-call/content suffix in the template's reasoning form.
            single = deepcopy(ms[i])
            single["reasoning_content"] = reasoning
            own = native(tokenizer, [{"role": "user", "content": ""}, single], [])
            body = own[own.index(HEADER) + len(HEADER):own.rindex(END)]
        elif inserted:
            body = EMPTY + body
        think_end = body.index("\n</think>\n\n") + len("\n</think>\n\n")
        rstart = len("<think>\n")
        replacements.append(dict(message_index=i, left=start, right=stop, body=body,
            inserted=inserted, source_reasoning=source_reasoning,
            reasoning=[rstart, rstart + len(reasoning)], content=[think_end, len(body)]))
    text, cursor, shift = "", 0, 0
    steps, details = [], []
    for change in replacements:
        text += base[cursor:change["left"]]
        origin = len(text)
        text += change["body"]
        reason = [origin + v for v in change["reasoning"]]
        content = [origin + v for v in change["content"]]
        left = reason[0] if reason[0] != reason[1] else content[0]
        steps.append([left, content[1]])
        details.append({"message_index": change["message_index"], "think_block_inserted": change["inserted"],
            "has_source_reasoning": change["source_reasoning"], "reasoning_char_span": reason,
            "content_char_span": content, "expected_text": change["body"][left-origin:]})
        delta = len(change["body"]) - (change["right"]-change["left"])
        for c in context:
            if c["char_span"][0] >= change["right"] + shift:
                c["char_span"] = [v + delta for v in c["char_span"]]
        shift += delta
        cursor = change["right"]
    text += base[cursor:]
    if row["render_mode"] == "transcript":
        start = details[0]["content_char_span"][0]
        end = details[0]["content_char_span"][1]
        if text[start:end] != ms[2]["content"]:
            raise ValueError("transcript content changed by template")
        steps = [[start+a, start+b] for a, b in row["entry_spans"]]
        details = [{"entry_index": i, "message_index": 2,
                    "think_block_inserted": details[0]["think_block_inserted"],
                    "has_source_reasoning": False, "reasoning_char_span": [a, a],
                    "content_char_span": [a, b], "expected_text": ms[2]["content"][x:y],
                    "field_char_spans": {k: [start+u, start+v] for k, (u, v) in row["entry_field_spans"][i].items()}}
                   for i, ((a, b), (x, y)) in enumerate(zip(steps, row["entry_spans"]))]
    for (a, b), d in zip(steps, details):
        if text[a:b] != d["expected_text"]:
            raise ValueError("rendered step content mismatch")
    return text, {"step_char_spans": steps, "steps": details, "context_spans": context}


def tokenize(item, tokenizer, *, max_context=32768, check=True):
    row = validate(item)
    text, spans = render(row, tokenizer)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, verbose=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    # First rendered message is conditioning only. Everything after it is teacher-forced.
    boundary = text.index(END) + len(END)
    first_after = next((i for i, (a, b) in enumerate(offsets) if a >= boundary), len(ids))
    if first_after == 0 or not any(b == boundary for a, b in offsets[:first_after]):
        raise ValueError(f"{row['uid']}: first-message token boundary")
    starts = [a for a, b in offsets]
    ends = [b for a, b in offsets]

    def locate(chars, exact=False):
        a, b = chars
        if a == b:
            i = bisect_left(starts, a)
            return [i-first_after, i-first_after]
        begin, end = bisect_right(ends, a), bisect_left(starts, b)
        if begin >= end or offsets[begin][0] > a or offsets[end-1][1] < b:
            raise ValueError(f"{row['uid']}: unmapped span {chars}")
        if exact:
            # A span maps to the minimal contiguous tokens that cover it (the same rule as
            # external_generalization.contracts.tokenize_answer). Qwen3 BPE may merge the
            # last char of a span with following newlines (".\n\n") or the first char with
            # preceding newlines; such a boundary token belongs to this span. Allowed only if
            # (1) the covered tokens decode exactly to the covered text, and (2) the overhang
            # outside [a, b) is whitespace. A token shared by two steps is rejected below.
            lo, hi = offsets[begin][0], offsets[end-1][1]
            if tokenizer.decode(ids[begin:end], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != text[lo:hi]:
                raise ValueError(f"{row['uid']}: decoded tokens differ at span {chars}")
            if text[lo:a].strip() or text[b:hi].strip():
                raise ValueError(f"{row['uid']}: non-whitespace token overhang at span {chars}")
            overhang[0] += (a - lo) + (hi - b)
            overhang[1] += int(lo < a or hi > b)
        return [begin-first_after, end-first_after]

    overhang = [0, 0]

    token_spans = [locate(c, exact=check) for c in spans["step_char_spans"]]
    if any(b > c for (a, b), (c, d) in zip(token_spans, token_spans[1:])):
        raise ValueError(f"{row['uid']}: one token crosses two steps")
    details = []
    for d in spans["steps"]:
        details.append({**{k:v for k,v in d.items() if k != "expected_text"},
            "reasoning_token_span": locate(d["reasoning_char_span"]),
            "content_token_span": locate(d["content_char_span"]),
            "field_token_spans": {k:locate(v) for k,v in d.get("field_char_spans", {}).items()}})
    return {"uid": row["uid"], "benchmark": row["benchmark"], "subset": row["subset"],
            "source_id": row["source_id"], "group": row["group"], "render_mode": row["render_mode"],
            "step_message_indices": row["step_message_indices"], "step_entry_indices": row["step_entry_indices"],
            "entry_to_history": row["entry_to_history"],
            "prompt_ids": ids[:first_after], "gen_ids": ids[first_after:], "full_text": text,
            "token_offsets": offsets, "gen_token_offsets": offsets[first_after:],
            "step_char_spans": spans["step_char_spans"], "step_token_spans": token_spans,
            "context_spans": [{**c, "token_span": locate(c["char_span"])} for c in spans["context_spans"]],
            "step_details": details, "overlength": len(ids)>max_context,
            "boundary_overhang": {"chars": overhang[0], "spans": overhang[1]},
            "span_coordinate_system": "chars=full_text; tokens=gen_ids; negative context indices are prompt",
            "first_message_token_count": first_after}
