"""
TinyCardioUNet compression: Tucker-2 (Conv1d encoder/decoder) + SVD low-rank
(the graph module's Linear layers), with automatic rank selection via
global-analytic VBMF (Nakajima 2013), following the one-shot pipeline of
Kim et al. (2016, ICLR):
    (1) VBMF rank selection -> (2) Tucker/SVD decomposition -> (3) fine-tuning

The graph module (src.model.BottleneckGraphSAGE, model.graph) uses plain
nn.Linear for its self/neighbor transforms, so svd_linear applies to it
directly like any other Linear layer.

Requires: src/vbmf.py (EVBMF).
Verified with torch 2.13, tensorly 0.9.0, scipy.

References
----------
Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, and Dongjun Shin, “Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications,” in 4th International Conference on Learning Representations (ICLR), 2016.
"""
import torch
import torch.nn as nn
import tensorly as tl
from tensorly import unfold
from tensorly.decomposition import partial_tucker

from src.vbmf import EVBMF

tl.set_backend("pytorch")


# ---------------------------------------------------------------- rank selection
def vbmf_ranks_conv1d(layer, min_rank=1, cap=0.9):
    """VBMF ranks (r_in, r_out) for the two channel modes of Conv1d weight (T,S,d).

    cap: never keep more than `cap * full` channels (guards against no-compression).
    """
    W = layer.weight.data
    T, S, _ = W.shape
    unfold_out = unfold(W, 0).cpu().numpy()   # (T, S*d)
    unfold_in = unfold(W, 1).cpu().numpy()    # (S, T*d)
    _, S_out, _, _ = EVBMF(unfold_out)
    _, S_in, _, _ = EVBMF(unfold_in)
    r_out = min(max(int(S_out.shape[0]), min_rank), int(cap * T))
    r_in = min(max(int(S_in.shape[0]), min_rank), int(cap * S))
    return r_in, r_out


def vbmf_rank_linear(layer, min_rank=1, cap=0.9):
    """VBMF rank for a Linear weight matrix (out, in)."""
    W = layer.weight.data.cpu().numpy()
    _, Sg, _, _ = EVBMF(W)
    full = min(W.shape)
    return min(max(int(Sg.shape[0]), min_rank), int(cap * full))


# ---------------------------------------------------------------- decompositions
def tucker2_conv1d(layer, r_in, r_out):
    """Conv1d(S,T,d) -> Conv1d(S,r_in,1) -> Conv1d(r_in,r_out,d) -> Conv1d(r_out,T,1)."""
    W = layer.weight.data
    T, S, d = W.shape
    (core, factors), _ = partial_tucker(W, rank=[r_out, r_in], modes=[0, 1], init="svd")
    f_out, f_in = factors

    first = nn.Conv1d(S, r_in, 1, bias=False)
    first.weight.data = f_in.t().unsqueeze(-1).contiguous()

    core_conv = nn.Conv1d(r_in, r_out, d, stride=layer.stride,
                          padding=layer.padding, bias=False)
    core_conv.weight.data = core.contiguous()

    last = nn.Conv1d(r_out, T, 1, bias=layer.bias is not None)
    last.weight.data = f_out.unsqueeze(-1).contiguous()
    if layer.bias is not None:
        last.bias.data = layer.bias.data.clone()

    return nn.Sequential(first, core_conv, last)


def svd_linear(layer, rank):
    """Linear(in,out) -> Linear(in,rank) -> Linear(rank,out)."""
    W = layer.weight.data
    out, inn = W.shape
    U, Sg, Vh = torch.linalg.svd(W, full_matrices=False)
    U, Sg, Vh = U[:, :rank], Sg[:rank], Vh[:rank, :]
    l1 = nn.Linear(inn, rank, bias=False)
    l1.weight.data = Vh.contiguous()
    l2 = nn.Linear(rank, out, bias=layer.bias is not None)
    l2.weight.data = (U * Sg.unsqueeze(0)).contiguous()
    if layer.bias is not None:
        l2.bias.data = layer.bias.data.clone()
    return nn.Sequential(l1, l2)


# ---------------------------------------------------------------- driver
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def _get(model, path):
    parent = model
    *parents, name = path.split(".")
    for p in parents:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, name


def _set(parent, name, new):
    if name.isdigit():
        parent[int(name)] = new
    else:
        setattr(parent, name, new)


def decompose_tinycardiounet_vbmf(model, conv_paths=None, graph_cap=0.9, verbose=True):
    """Auto-rank compression. conv_paths: list of Conv1d module paths to compress.

    Only channel-heavy layers are worth compressing; enc1 (in=3) and dec1 (1x1)
    are skipped by default.
    """
    if conv_paths is None:
        conv_paths = ["enc4.conv.0", "dec4_up.up.1", "dec4_conv.conv.0",
                      "dec3_conv.conv.0", "dec2_conv.conv.0"]

    for path in conv_paths:
        parent, name = _get(model, path)
        layer = parent[int(name)] if name.isdigit() else getattr(parent, name)
        r_in, r_out = vbmf_ranks_conv1d(layer)
        before = n_params(layer)
        new = tucker2_conv1d(layer, r_in, r_out)
        _set(parent, name, new)
        if verbose:
            print(f"  {path:20s} in={layer.in_channels}->{r_in}, "
                  f"out={layer.out_channels}->{r_out}  "
                  f"params {before}->{n_params(new)}")

    # BottleneckGraphSAGE: two Linear layers (lin_self, lin_neighbor).
    if hasattr(model, "graph"):
        for attr in ("lin_self", "lin_neighbor"):
            lin = getattr(model.graph, attr, None)
            if isinstance(lin, nn.Linear):
                r = vbmf_rank_linear(lin, cap=graph_cap)
                before = n_params(lin)
                new = svd_linear(lin, r)
                setattr(model.graph, attr, new)
                if verbose:
                    print(f"  graph.{attr:14s} rank -> {r}  params {before}->{n_params(new)}")

    return model
