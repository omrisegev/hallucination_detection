# Why this acquisition uses K=512, not 4096

Written 2026-08-18, at acquisition launch, before any result was opened.

## The decision

The DeepConf pool here is **30 AIME24 questions x 512 traces = 15,360**, not the
30 x 4,096 = 122,880 originally planned.

## Why

The K=4096 run (kept intact at `../m2_deepconf_full`, 12,370 traces) was measured at
1,180 traces/hour across 24 shards after 9.75 hours. The remaining 111,360 traces needed
roughly 94 further GPU-hours. The project sits against a hard purchased GPU-hour cap
(5,760 h, ~4,226 h already consumed), and the submitted chain provided about 38 hours.
That path ended at roughly 46% of the pool, and — because the driver strides units in
question-major order — at only about a third of the 30 questions, with the rest holding
no traces at all.

K=512 covers **all 30 questions** and finishes in roughly 13 hours.

## What it costs, precisely

The largest online budget in our frozen registry is 512, and DeepConf's Algorithm 2
budgets are (32, 64, 128, 256, 512). So **every preregistered budget remains runnable at
full width**. What is out of reach is a full-size offline majority vote over a
4,096-trace pool, which is the paper's largest offline configuration.

This is recorded as a `declared_deviation` on the field `traces_per_question_K`, and the
run's fidelity stays `paper-specified-partial`.

## Why it is not a one-way door

Resume is by stable trace key (`<question_id>#<trace_index>`), so raising K later is
purely additive: trace indices 512..N would simply be appended, and nothing already
acquired is redone.

## Why a new directory

Sharding is `i % n_shards` over `i = question_index * K + trace_index`. Changing K
changes the modulus (4096 mod 24 = 16 versus 512 mod 24 = 8), so trace ownership moves
between shards for every question after the first. Resuming into the old directory would
have made one worker regenerate keys another worker already held — cross-worker duplicate
keys, the exact invariant the `part_NN` layout exists to protect. A separate directory
removes the hazard entirely, at a cost of about 1,536 regenerated traces.

## The predecessor is kept

`../m2_deepconf_full` is not deleted. It is valid, hash-verified data and remains a
deeper K=4096 pool for the two questions it completed (2024-I-1 and 2024-I-10).
