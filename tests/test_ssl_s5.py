"""Plan section 17.2 tests for S5-CPU, run BEFORE real-data scoring."""
import numpy as np
import pytest
import torch
from scipy.special import log_softmax, expit

from spectral_utils import ssl_s5 as S5

rng = np.random.default_rng(0)


def _segments(sizes):
    seg = torch.tensor(np.repeat(np.arange(len(sizes)), sizes), dtype=torch.long)
    return seg, len(sizes)


def test_per_answer_per_channel_standardization():
    x = rng.normal(3., 7., size=(40, 11)); x[:, 4] = 2.5                      # constant channel
    z = S5.standardize_tokens_answer(x)
    assert np.allclose(z.mean(0), 0, atol=1e-12)
    ok = [c for c in range(11) if c != 4]
    assert np.allclose(z[:, ok].std(0), 1, atol=1e-12) and np.all(z[:, 4] == 0)
    heavy = np.concatenate([rng.normal(size=(99, 1)), [[1e6]]])               # the S2 explosion case
    assert abs(S5.standardize_tokens_answer(heavy).std()) - 1 < 1e-9


def test_segment_ops_match_a_reference_loop():
    sizes = [1, 5, 3, 7]; seg, nseg = _segments(sizes)
    e = torch.tensor(rng.normal(0, 40, size=sum(sizes)))                      # large values: max-shift must hold
    a = S5.segment_softmax(e, seg, nseg).numpy(); lse = S5.segment_logsumexp(e, seg, nseg).numpy()
    pos = 0
    for j, s in enumerate(sizes):
        v = e[pos:pos + s].numpy()
        assert np.allclose(a[pos:pos + s], np.exp(log_softmax(v)), atol=1e-12)
        assert abs(lse[j] - (np.log(np.exp(v - v.max()).sum()) + v.max())) < 1e-10
        pos += s
    assert np.allclose(S5.segment_sum(torch.ones(sum(sizes), dtype=torch.float64), seg, nseg).numpy(), sizes)


def test_attention_with_flat_logits_equals_mean_pooling():
    sizes = [4, 1, 6]; seg, nseg = _segments(sizes)
    f = torch.tensor(rng.normal(size=(sum(sizes), 11)))
    torch.manual_seed(0); att = S5.StepHead(11, pooling='attention').double()
    torch.manual_seed(0); mean = S5.StepHead(11, pooling='mean').double()
    with torch.no_grad(): att.v.zero_()                                       # v = 0 -> flat logits
    ua, aa, _ = att(f, seg, nseg); um, am, _ = mean(f, seg, nseg)
    assert torch.allclose(att.risk.weight, mean.risk.weight)                  # same seed -> same risk head
    assert np.allclose(aa.detach().numpy(), am.detach().numpy(), atol=1e-14)
    assert np.allclose(ua.detach().numpy(), um.detach().numpy(), atol=1e-14)
    assert np.allclose(am.detach().numpy(), np.repeat([1 / s for s in sizes], sizes))


def test_losses_match_reference_and_one_step_pb_answer_is_free():
    sizes = [3, 1, 4]; sa, nans = _segments(sizes)                            # steps grouped by answer
    u = torch.tensor(rng.normal(size=sum(sizes)), requires_grad=True)
    q = np.concatenate([rng.dirichlet(np.ones(s)) for s in sizes]); qt = torch.tensor(q)
    wa = torch.tensor(rng.uniform(.5, 1.5, nans)); wa = wa / wa.sum()
    ref = 0.; pos = 0
    for j, s in enumerate(sizes):
        ref += float(wa[j]) * -float(q[pos:pos + s] @ log_softmax(u.detach().numpy()[pos:pos + s])); pos += s
    assert abs(float(S5.loss_pb(u, qt, sa, nans, wa)) - ref) < 1e-12
    one = torch.tensor([2.7], requires_grad=True); s1, n1 = _segments([1])
    assert abs(float(S5.loss_pb(one, torch.ones(1, dtype=torch.float64), s1, n1, torch.ones(1, dtype=torch.float64)))) < 1e-15
    qb = torch.tensor(rng.uniform(0, 1, sum(sizes)))
    ref = 0.; pos = 0
    for j, s in enumerate(sizes):
        uu = u.detach().numpy()[pos:pos + s]; tt = qb.numpy()[pos:pos + s]
        bce = np.maximum(uu, 0) - uu * tt + np.log1p(np.exp(-np.abs(uu)))
        ref += float(wa[j]) * bce.mean(); pos += s
    assert abs(float(S5.loss_prm(u, qb, sa, nans, wa)) - ref) < 1e-12
    assert np.allclose(expit(u.detach().numpy())[:0], [])


def test_fit_rejects_labels_and_freezes_arbitrary_attention():
    sizes = [4, 5, 3, 2]; seg, nseg = _segments(sizes); sa, nans = _segments([2, 2])
    f = torch.tensor(rng.normal(size=(sum(sizes), 11)), dtype=torch.float32)
    q = torch.tensor(np.concatenate([rng.dirichlet(np.ones(2)), rng.dirichlet(np.ones(2))]), dtype=torch.float32)
    wa = torch.full((2,), .5)
    def batches(_): return f, seg, nseg, q, sa, nans, wa
    with pytest.raises(ValueError, match='label-like'):
        S5.train_head(batches, 'pb_q4', 11, seed=0, updates=1, labels=np.zeros(2))
    m0 = S5.StepHead(11)
    frozen, _ = S5.train_head(batches, 'pb_q4', 11, train_attention=False, seed=0, updates=25)
    torch.manual_seed(0); init = S5.StepHead(11, train_attention=False)
    assert torch.allclose(frozen.proj.weight, init.proj.weight) and torch.allclose(frozen.v, init.v)
    assert not torch.allclose(frozen.risk.weight, init.risk.weight)           # the risk head still moved
    learned, _ = S5.train_head(batches, 'pb_q4', 11, train_attention=True, seed=0, updates=25)
    assert not torch.allclose(learned.v, init.v)
    again, _ = S5.train_head(batches, 'pb_q4', 11, train_attention=True, seed=0, updates=25)
    assert torch.allclose(learned.v, again.v) and torch.allclose(learned.risk.weight, again.risk.weight)
    other, _ = S5.train_head(batches, 'pb_q4', 11, train_attention=True, seed=1, updates=25)
    assert not torch.allclose(learned.v, other.v)
    assert isinstance(m0, S5.StepHead)


def test_weighted_scale_matches_reference_and_is_label_free():
    u = rng.normal(2., 3., 500); w = rng.uniform(.1, 2., 500)
    mu, sd = S5.weighted_mean_sd(u, w); wn = w / w.sum()
    assert abs(mu - wn @ u) < 1e-12 and abs(sd - np.sqrt(wn @ (u - mu) ** 2)) < 1e-12
    s = S5.standardize_scores(u, mu, sd)
    assert abs(wn @ s) < 1e-9 and abs(np.sqrt(wn @ (s - wn @ s) ** 2) - 1) < 1e-9
    assert np.allclose(S5.standardize_scores(u, mu, 0.), (u - mu) / 1e-8)     # degenerate SD floor
