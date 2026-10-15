"""In-depth Linear CEC parameter behavioral tests."""

from __future__ import annotations

import torch

from bumblebee.param_sweep import probe_behavioral


def test_probe_behavioral_all_pass():
    report = probe_behavioral(seed=0)
    assert report["all_pass"], {k: v for k, v in report.items() if isinstance(v, dict) and not v.get("pass", True)}


def test_mass_fn_topk_support_bound():
    from bumblebee.model import SoftCorrelator

    corr = SoftCorrelator(dim=8, mass_fn="topk", topk=2, learn_temperature=False)
    mass, _ = corr(torch.randn(1, 8), torch.randn(1, 12, 8), torch.ones(1, 12))
    assert int((mass[0] > 1e-6).sum()) <= 2


def test_temperature_softens_softmax():
    from bumblebee.model import SoftCorrelator

    q, p, m = torch.randn(1, 8), torch.randn(1, 10, 8), torch.ones(1, 10)
    soft = SoftCorrelator(dim=8, mass_fn="softmax", learn_temperature=False, temperature=0.25)
    hardish = SoftCorrelator(dim=8, mass_fn="softmax", learn_temperature=False, temperature=4.0)
    m_s, _ = soft(q, p, m)
    m_h, _ = hardish(q, p, m)
    # lower tau → peakier (higher max)
    assert float(m_s.max()) >= float(m_h.max()) - 1e-5


def test_gated_delta_max_finite_under_stress():
    from bumblebee.model import GatedDeltaCorrelator

    g = GatedDeltaCorrelator(dim=8, learn_temperature=False, delta_max=0.5)
    with torch.no_grad():
        g.delta_proj.weight.mul_(10)
        g.delta_proj.bias.fill_(5)
    mass, ev = g(torch.randn(2, 8), torch.randn(2, 9, 8), torch.ones(2, 9))
    assert torch.isfinite(mass).all() and torch.isfinite(ev).all()


def test_ssm_delta_max_finite_under_stress():
    from bumblebee.model import SelectiveStateEncoder

    enc = SelectiveStateEncoder(32, dim=8, pad_id=0, delta_max=0.5)
    with torch.no_grad():
        enc.delta_proj.weight.mul_(10)
        enc.delta_proj.bias.fill_(5)
    out = enc(torch.randint(1, 32, (2, 7)))
    assert torch.isfinite(out).all()


def test_distill_lambda_mass_isolates_mass_kl():
    from bumblebee.distill import distill_loss
    from bumblebee.model import CECModel

    t = CECModel(30, dim=8, encoder_kind="gru")
    s = CECModel(30, dim=8, encoder_kind="ssm", correlator_kind="gated_delta")
    for p in t.parameters():
        p.requires_grad_(False)
    q = torch.randint(1, 30, (2, 3))
    p = torch.randint(1, 30, (2, 5))
    mask = (p != 0).float()
    with torch.no_grad():
        to = t(q, p)
    so = s(q, p)
    loss, met = distill_loss(
        to, so, passage_mask=mask, lambda_mass=1.0, lambda_span=0.0, lambda_gate=0.0, lambda_task=0.0
    )
    assert abs(float(loss.item()) - met["mass_kl"]) < 1e-4


def test_cec_topk_and_delta_kwargs_wire():
    from bumblebee.model import CECModel, GatedDeltaCorrelator, SelectiveStateEncoder

    m = CECModel(
        40,
        dim=12,
        encoder_kind="ssm",
        correlator_kind="gated_delta",
        mass_fn="topk",
        topk=3,
        encoder_delta_max=1.5,
        correlator_delta_max=0.75,
        learn_temperature=False,
        temperature=2.0,
    )
    assert isinstance(m.encoder, SelectiveStateEncoder)
    assert isinstance(m.correlator, GatedDeltaCorrelator)
    assert m.encoder.delta_max == 1.5
    assert m.correlator.delta_max == 0.75
    assert m.correlator.topk == 3
    assert abs(m.correlator.temperature - 2.0) < 1e-3
