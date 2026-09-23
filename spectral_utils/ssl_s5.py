"""S5-CPU of the SSL / pseudo-label / residual localization plan (v1.1, section 11), reformulated
without the S4 self-supervised encoder.

The plan's S5 concatenates three token blocks -- 11 telemetry channels, 11 masked-linear residuals
and a 32-dim frozen SSL embedding -- and learns an attention pooling inside each step.  The last two
blocks come from the S4 GPU stage, which Omri excluded on 2026-09-23.  What survives without a GPU
is exactly the plan's own baseline arm `H_RAW`: learned attention pooling over the 11 raw token
channels, with the SAME target (P_SOFT teacher), the same role contract and the same endpoints.

Representation fix carried over from S2: the token matrix is standardized PER ANSWER and PER CHANNEL.
The frozen median/IQR normalization of `cvf_v2.core.profiles` leaves top50_js, chosen_surprisal and
true_tail50 with variance 1e6-3e9, so any pooling that mixes channels is otherwise dominated by them.

No correctness label, error index, error type or gate enters any fit here.
"""
import numpy as np
import torch

from .ssl_s1 import CHANNELS   # noqa: F401  (re-exported: the 11 frozen channels, same order)

HIDDEN = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
UPDATES = 2000
BATCH_ANSWERS = 16
SEEDS = (0, 1, 2)
NEG_INF = -1e30

BANNED = {'labels', 'label', 'y', 'y_true', 'target_step', 'error_steps', 'first_error',
          'correct', 'classification', 'gate'}


def _check_no_labels(kwargs):
    bad = BANNED & set(kwargs)
    if bad:
        raise ValueError(f'S5 fit does not accept label-like inputs: {sorted(bad)}')


# ------------------------------------------------------------------ representation
def standardize_tokens_answer(x):
    """Per-answer, per-channel standardization of one answer's raw token matrix.

    Mirrors the teacher's step-profile rule (ssl_s1.answer_z): subtract the answer's channel mean,
    divide by its channel SD, SD <= 1e-8 -> 1 (a constant channel becomes all zeros)."""
    x = np.asarray(x, float)
    sd = x.std(0)
    return (x - x.mean(0)) / np.where(sd > 1e-8, sd, 1.)


# ------------------------------------------------------------------ segment ops
def segment_softmax(e, seg, nseg):
    """softmax of `e` inside each segment id in `seg` (0..nseg-1). Max-shift is detached, so the
    result is mathematically identical to a plain per-segment softmax."""
    m = torch.full((nseg,), NEG_INF, dtype=e.dtype).scatter_reduce(0, seg, e.detach(), 'amax', include_self=True)
    ex = torch.exp(e - m[seg])
    den = torch.zeros(nseg, dtype=e.dtype).index_add(0, seg, ex)
    return ex / den[seg]


def segment_sum(x, seg, nseg):
    return torch.zeros(nseg, dtype=x.dtype).index_add(0, seg, x)


def segment_logsumexp(u, seg, nseg):
    m = torch.full((nseg,), NEG_INF, dtype=u.dtype).scatter_reduce(0, seg, u.detach(), 'amax', include_self=True)
    return m + torch.log(segment_sum(torch.exp(u - m[seg]), seg, nseg))


# ------------------------------------------------------------------ head
class StepHead(torch.nn.Module):
    """u_s = sum_{t in step s} attention_t * (w^T f_t + b).

    pooling='attention': attention_t = softmax_within_step(v^T tanh(W f_t + a)), hidden_dim=32.
    pooling='mean':      attention_t = 1 / n_tokens_s  (plan's H_SSL_MEAN control).
    train_attention=False freezes W, a, v at their random initialization (arbitrary attention of the
    same form), so the learned-vs-arbitrary attention contrast is isolated from capacity."""

    def __init__(self, d, hidden=HIDDEN, pooling='attention', train_attention=True):
        super().__init__()
        self.pooling = pooling
        self.risk = torch.nn.Linear(d, 1)
        if pooling == 'attention':
            self.proj = torch.nn.Linear(d, hidden)
            bound = 1.0 / np.sqrt(hidden)
            self.v = torch.nn.Parameter(torch.empty(hidden).uniform_(-bound, bound))
            if not train_attention:
                for p in list(self.proj.parameters()) + [self.v]:
                    p.requires_grad_(False)
        elif pooling != 'mean':
            raise ValueError(pooling)

    def attention(self, f, seg, nseg):
        if self.pooling == 'mean':
            cnt = segment_sum(torch.ones(len(f), dtype=f.dtype), seg, nseg)
            return 1.0 / cnt[seg]
        return segment_softmax(self.v @ torch.tanh(self.proj(f)).T, seg, nseg)

    def forward(self, f, seg, nseg):
        risk = self.risk(f).squeeze(-1)
        alpha = self.attention(f, seg, nseg)
        return segment_sum(alpha * risk, seg, nseg), alpha, risk


# ------------------------------------------------------------------ loss (plan 7.2 targets)
def loss_pb(u, q, step_answer, nans, w_ans):
    """-sum_a w_a sum_{s in a} q_s log_softmax_a(u)_s. A one-step answer contributes exactly 0."""
    lse = segment_logsumexp(u, step_answer, nans)
    return -(w_ans[step_answer] * q * (u - lse[step_answer])).sum()


def loss_prm(u, q, step_answer, nans, w_ans, m=None):
    """sum_a w_a * weighted mean over the answer's steps of BCEWithLogits(u_s, q_s)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(u, q, reduction='none')
    m = torch.ones_like(u) if m is None else m
    mass = segment_sum(m, step_answer, nans).clamp_min(1e-12)
    return (w_ans[step_answer] / mass[step_answer] * m * bce).sum()


# ------------------------------------------------------------------ weighted scale on B (plan 11)
def weighted_mean_sd(u, w):
    """Weighted mean/SD of the step scores; w is the step's share of its answer's mass (5.3)."""
    w = np.asarray(w, float); w = w / w.sum(); u = np.asarray(u, float)
    mu = float(w @ u)
    return mu, float(np.sqrt(max(w @ (u - mu) ** 2, 0.0)))


def standardize_scores(u, mu, sd):
    return (np.asarray(u, float) - mu) / max(sd, 1e-8)


# ------------------------------------------------------------------ training
def train_head(batches, task, d, pooling='attention', train_attention=True, seed=0,
               updates=UPDATES, lr=LR, weight_decay=WEIGHT_DECAY, **kwargs):
    """`batches` is a callable(update_index) -> (f, seg, nseg, q, step_answer, nans, w_ans) of torch
    tensors, drawn by the caller with the plan 5.3 answer weights.  AdamW, fixed number of updates,
    the LAST head is the result: no checkpoint selection, no validation on H/C, no labels."""
    _check_no_labels(kwargs)
    torch.manual_seed(seed)
    model = StepHead(d, pooling=pooling, train_attention=train_attention)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    fn = loss_pb if task.startswith('pb') else loss_prm
    trace = np.empty(updates)
    for it in range(updates):
        f, seg, nseg, q, sa, nans, wa = batches(it)
        u, _, _ = model(f, seg, nseg)
        loss = fn(u, q, sa, nans, wa)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        trace[it] = float(loss.detach())
    return model, trace


@torch.no_grad()
def score_answer(model, f, seg, nseg):
    u, alpha, risk = model(f, seg, nseg)
    return u.numpy(), alpha.numpy(), risk.numpy()
