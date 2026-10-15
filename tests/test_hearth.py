"""Hearth paging: shortlist numerics, disk bytes, and the RAM budget."""

import torch
import torch.nn.functional as F

from bumblebee.hearth import (
    ClusterCache,
    ClusterIndex,
    DecoderOnlyLM,
    HearthHead,
    PackedLinear,
    PagedLexicon,
    ResidentCore,
    ResidentLM,
    RowCache,
    SketchIndex,
    approximate_linear,
    attach_decoder,
    evidence_shortlist,
    pack_linears,
    shortlist_logits,
)
from bumblebee.model import GenerativeAnswerDecoder


def test_shortlist_matches_linear_columns(tmp_path):
    weight = torch.randn(40, 8)
    bias = torch.randn(40)
    hidden = torch.randn(8)
    ids = torch.tensor([3, 9, 11, 20])
    PagedLexicon.write(tmp_path / "rows.hrth", weight)
    lex = PagedLexicon(tmp_path / "rows.hrth")
    got = shortlist_logits(hidden, lex.read_ids(ids), bias, ids)
    expect = F.linear(hidden, weight, bias)[ids]
    assert torch.allclose(got, expect, atol=1e-5)


def test_repeat_step_reads_no_bytes(tmp_path):
    weight = torch.randn(200, 16)
    PagedLexicon.write(tmp_path / "rows.hrth", weight)
    lex = PagedLexicon(tmp_path / "rows.hrth")
    cache = RowCache(lex, budget_bytes=64 * 16 * 4, page_rows=1)
    ids = torch.arange(10)
    cache.rows(ids)
    assert lex.bytes_read == 10 * 16 * 4
    cache.rows(ids)
    assert lex.bytes_read == 10 * 16 * 4
    assert cache.resident_bytes <= 64 * 16 * 4


def test_cache_stays_inside_budget(tmp_path):
    weight = torch.randn(500, 8)
    PagedLexicon.write(tmp_path / "rows.hrth", weight)
    lex = PagedLexicon(tmp_path / "rows.hrth")
    budget = 4 * 8 * 4
    cache = RowCache(lex, budget_bytes=budget, page_rows=1)
    cache.rows(torch.arange(40))
    assert cache.resident_bytes <= budget
    assert lex.bytes_read < weight.numel() * 4


def test_int8_rows_roundtrip(tmp_path):
    weight = torch.randn(30, 12)
    PagedLexicon.write(tmp_path / "q.hrth", weight, dtype="int8")
    lex = PagedLexicon(tmp_path / "q.hrth")
    got = lex.read_range(0, 30)
    assert lex.bytes_read < weight.numel() * 4
    assert torch.allclose(got, weight, atol=0.05, rtol=0.05)


def test_cluster_gather_is_smaller_than_full_matrix(tmp_path):
    torch.manual_seed(0)
    weight = torch.randn(256, 16)
    index, packed = ClusterIndex.fit(weight, n_clusters=16, iters=2, seed=0)
    PagedLexicon.write(tmp_path / "head.hrth", packed)
    lex = PagedLexicon(tmp_path / "head.hrth")
    cache = ClusterCache(lex, index, budget_bytes=256 * 16 * 4)
    hidden = torch.randn(16)
    clusters = index.top_clusters(hidden, n_probe=2)
    rows, orig = cache.gather(clusters, cap=32)
    assert rows.size(0) <= 32
    assert orig.shape == rows.shape[:1]
    assert lex.bytes_read < weight.numel() * 4
    before = lex.bytes_read
    cache.gather(clusters, cap=32)
    assert lex.bytes_read == before


def test_full_cap_matches_dense_linear(tmp_path):
    torch.manual_seed(1)
    weight = torch.randn(48, 8)
    bias = torch.randn(48)
    hidden = torch.randn(8)
    index, packed = ClusterIndex.fit(weight, n_clusters=6, iters=2, seed=1)
    PagedLexicon.write(tmp_path / "w.hrth", packed)
    lex = PagedLexicon(tmp_path / "w.hrth")
    cache = ClusterCache(lex, index, budget_bytes=48 * 8 * 4)
    got = approximate_linear(hidden, index, cache, n_probe=6, cap=48, bias=bias, n_rows=48)
    expect = F.linear(hidden, weight, bias)
    assert torch.allclose(got, expect, atol=1e-4)


def test_evidence_shortlist_prefers_passage_ids():
    ids = evidence_shortlist([4, 4, 9], [1, 9, 2], cap=3)
    assert ids.tolist() == [4, 9, 1]


def test_resident_lm_generate_shape(tmp_path):
    dim, vocab = 8, 20
    core = ResidentCore(dim)
    emb = torch.randn(vocab, dim)
    out_w = torch.randn(vocab, dim)
    out_b = torch.randn(vocab)
    ResidentLM.export(tmp_path / "lm", emb, out_w, out_b, core)
    lm = ResidentLM.load(tmp_path / "lm", budget_bytes=32_000)
    ids = lm.generate(
        torch.randn(1, dim),
        torch.randn(1, dim),
        shortlist=torch.tensor([1, 2, 5, 7]),
        max_len=4,
    )
    assert ids.shape == (1, 4)
    assert set(ids[0].tolist()).issubset({1, 2, 5, 7})
    assert lm.out.lexicon.bytes_read < vocab * dim * 4 * 4


def test_decoder_exact_cap_matches_dense_token(tmp_path):
    torch.manual_seed(3)
    model = DecoderOnlyLM(vocab=32, dim=16, n_layers=1, n_heads=4, max_seq=8)
    model.eval()
    runtime = attach_decoder(
        model,
        tmp_path / "dec",
        budget_bytes=256_000,
        sketch_rank=8,
        cap=32,
    )
    idx = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        dense = int(model.forward(idx)[0, -1].argmax())
        hearth = runtime.next_id(runtime.backbone(idx)[0, -1])
    assert hearth == dense


def test_low_rank_shortlist_matches_dense_argmax():
    torch.manual_seed(5)
    weight = torch.randn(180, 8) @ torch.randn(8, 24)
    bias = torch.randn(180)
    sketch = SketchIndex.fit(weight, rank=8)
    for _ in range(6):
        hidden = torch.randn(24)
        true = int((weight @ hidden + bias).argmax())
        assert sketch.rank_of(hidden, true, bias) == 0
        ids = sketch.top_ids(hidden, 4, bias)
        pred = int(ids[int((weight.index_select(0, ids) @ hidden + bias.index_select(0, ids)).argmax())])
        assert pred == true


def test_full_rank_head_matches_when_a_short_list_would_miss():
    torch.manual_seed(0)
    weight = torch.randn(128, 32)
    bias = torch.randn(128)
    head = HearthHead(weight, bias, sketch_rank=8)
    assert head.exact_only
    for _ in range(8):
        hidden = torch.randn(32)
        assert int(head.argmax(hidden)) == int((weight @ hidden + bias).argmax())
        assert head.used_shortlist is False
        assert head.seal is not None and head.seal.kind == "dense"


def test_low_rank_head_certifies_and_matches():
    torch.manual_seed(1)
    weight = torch.randn(200, 6) @ torch.randn(6, 24)
    bias = torch.zeros(200)
    head = HearthHead(weight, bias, sketch_rank=6)
    assert head.exact_only is False
    for _ in range(6):
        hidden = torch.randn(24)
        assert int(head.argmax(hidden)) == int((weight @ hidden + bias).argmax())
        assert head.used_shortlist is True
        assert head.rows_scored < 200
        assert head.seal is not None and head.seal.kind == "sketch"
        assert head.seal.token == int(head.argmax(hidden))


def test_answer_decoder_hearth_matches_dense_tokens():
    torch.manual_seed(0)
    dec = GenerativeAnswerDecoder(40, dim=16)
    dec.eval()
    query = torch.randn(2, 16)
    evidence = torch.randn(2, 16)
    plain = dec.generate(query, evidence, max_len=5)
    dec.enable_hearth()
    assert dec.hearth is not None
    got = dec.generate(query, evidence, max_len=5)
    assert torch.equal(got, plain)


def test_packed_linear_matches_dense_and_is_smaller():
    torch.manual_seed(0)
    weight = torch.randn(64, 128)
    weight[:, 3] *= 20
    bias = torch.randn(64)
    hidden = torch.randn(4, 128)
    packed = PackedLinear(weight, bias, outlier_fraction=0.05)
    got = packed(hidden)
    expect = torch.nn.functional.linear(hidden, weight, bias)
    assert torch.nn.functional.cosine_similarity(got, expect, dim=-1).min() > 0.99
    assert packed.nbytes < weight.numel() * 4 / 2
    assert int(packed.outlier_idx.numel()) >= 1


def test_pack_linears_leaves_the_head_exact():
    torch.manual_seed(0)
    model = DecoderOnlyLM(vocab=40, dim=32, n_layers=1, n_heads=4, max_seq=8)
    head = model.lm_head.weight.detach().clone()
    n = pack_linears(model, outlier_fraction=0.05)
    assert n >= 1
    assert torch.equal(model.lm_head.weight, head)
    idx = torch.randint(0, 40, (2, 4))
    out = model(idx)
    assert out.shape == (2, 4, 40)


def test_sketch_finds_a_clear_winner():
    torch.manual_seed(4)
    weight = torch.randn(80, 16) * 0.05
    hidden = torch.randn(16)
    weight[7] = hidden * 4
    sketch = SketchIndex.fit(weight, rank=8)
    top = sketch.top_ids(hidden, k=4)
    assert int(top[0]) == 7 or 7 in top.tolist()


def test_i8_gemv_matches_quantized_dot():
    torch.manual_seed(0)
    weight = torch.randn(200, 40)
    scale = weight.abs().amax(1).clamp(min=1e-8) / 127
    quant = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    hidden = torch.randn(40)
    from bumblebee.hearth import _i8_gemv

    got = _i8_gemv(quant, scale.contiguous(), None, hidden)
    expect = (quant.float() @ hidden) * scale
    assert (got - expect).abs().max() < 1e-3


def test_quant_head_matches_dense_and_checks_few_rows():
    torch.manual_seed(1)
    n_rows, dim = 2048, 48
    codes = torch.randint(-30, 31, (n_rows, dim))
    scale = torch.empty(n_rows).uniform_(0.002, 0.02)
    weight = codes.float() * scale[:, None]
    bias = torch.randn(n_rows) * 0.01
    head = HearthHead(weight, bias)
    assert head.quant_ready
    hidden = torch.randn(6, dim)
    hidden[0] = weight[3]
    got = head.argmax(hidden)
    expect = torch.nn.functional.linear(hidden, weight, bias).argmax(dim=-1)
    assert torch.equal(got, expect)
    head.argmax(hidden[0])
    assert head.used_quant
    assert head.rows_scored < 64


def test_random_quant_head_still_matches_dense():
    torch.manual_seed(2)
    weight = torch.randn(1200, 32)
    bias = torch.randn(1200)
    head = HearthHead(weight, bias)
    hidden = torch.randn(4, 32)
    got = head.argmax(hidden)
    expect = torch.nn.functional.linear(hidden, weight, bias).argmax(dim=-1)
    assert torch.equal(got, expect)
    assert len(head.seals) == 4
    assert all(seal.kind in ("pulse", "dense") for seal in head.seals)
    assert head.seal is not None and head.seal.token == int(got[-1])


def test_pulse_follows_this_hidden_state_and_matches_dense():
    torch.manual_seed(3)
    n_rows, dim = 2048, 64
    weight = torch.randn(n_rows, dim) * 0.02
    bias = torch.randn(n_rows) * 0.01
    head = HearthHead(weight, bias)
    assert head.quant_ready
    hidden = torch.zeros(dim)
    hidden[2] = 4.0
    hidden[9] = -3.5
    hidden[40] = 1.2
    hidden = hidden + torch.randn(dim) * 0.01
    got = int(head.argmax(hidden))
    expect = int(torch.nn.functional.linear(hidden, weight, bias).argmax())
    assert got == expect
    assert head.seal is not None
    assert head.seal.kind == "pulse"
    assert head.seal.pulse == min(32, max(4, dim // 16))
    assert head.seal.token == got
    assert head.seal.live_rows == head.rows_scored
    assert head.rows_scored < n_rows


def test_decoder_generate_records_a_seal_per_word():
    torch.manual_seed(4)
    dec = GenerativeAnswerDecoder(1200, dim=32)
    dec.eval()
    other = GenerativeAnswerDecoder(1200, dim=32)
    other.load_state_dict(dec.state_dict())
    other.eval()
    dec.enable_hearth()
    query = torch.randn(1, 32)
    evidence = torch.randn(1, 32)
    hearth_ids = dec.generate(query, evidence, max_len=3)
    dense_ids = other.generate(query, evidence, max_len=3)
    assert torch.equal(hearth_ids, dense_ids)
    assert len(dec.hearth_seals) == 3
    assert all(seal.kind in ("pulse", "dense") for seal in dec.hearth_seals)
    assert [seal.token for seal in dec.hearth_seals] == [int(t) for t in hearth_ids[0]]
