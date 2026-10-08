#!/usr/bin/env python3
"""
Stuck, Wang & Naud (2025) — SIGNED Burstprop, symmetric feedback,
strict-online learning accelerated with CUDA Graph chunks.

Why this version exists
-----------------------
The paper trains samples sequentially. A normal minibatch would change the
algorithm because all samples in a minibatch would see the same weights.

This implementation keeps the exact online dependency:

    sample 1 -> update W
    sample 2 -> sees updated W -> update W
    ...
    sample K -> sees all previous updates

but captures K sequential samples x 100 ms into ONE CUDA graph. Therefore
"chunk_size" is an execution chunk, NOT a learning minibatch.

The default initialization is a discrete fluctuation-driven initialization
adapted to this exact LIF update:

    V[t+1] = a V[t] + sum_j w_j S_j[t],  a = 1 - dt/tau

For independent Bernoulli/Poisson events with probability p per time step and
zero-mean weights:

    Var[V] ~= n * sigma_w^2 * p(1-p) / (1-a^2)

so targeting membrane fluctuation sigma_U gives

    sigma_w = sigma_U * sqrt((1-a^2) / (n*p*(1-p)))

The default sigma_U=0.5 corresponds to threshold distance xi≈2 when threshold=1
and mean membrane potential≈0, inside the fluctuation-driven regime advocated
by Rossbroich, Gygax & Zenke (2022).

IMPORTANT:
The exact numerical initialization prefactor used by Stuck et al. (2025) is not
reported. Thus --init-mode fluctuation is a principled reproduction choice, not
a claim that sigma_U=0.5 is the authors' hidden exact value.

No autograd, optimizer, surrogate gradient, BPTT, or minibatch-gradient update.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchvision import datasets


@dataclass
class Config:
    # Architecture
    n_input: int = 784
    n_hidden: int = 100
    n_output: int = 10

    # Dynamics
    dt_ms: float = 1.0
    presentation_ms: float = 100.0
    tau_s_ms: float = 10.0
    tau_d_ms: float = 10.0
    tau_elig_ms: float = 10.0
    tau_rate_ms: float = 10.0
    v_threshold: float = 1.0
    v_reset: float = 0.0

    # Signed Burstprop
    eta: float = 1e-4
    target_high_hz: float = 200.0
    target_low_hz: float = 20.0
    hidden_target_hz: float = 20.0
    gamma_teacher: float = 1e-3
    gamma_hidden: float = 1e-5

    # Input
    input_min_hz: float = 20.0
    input_max_hz: float = 200.0
    image_mean_target: Optional[float] = None

    # Initialization
    init_mode: str = "fluctuation"  # fluctuation | sqrt
    sigma_u_target: float = 0.5
    init_gain: float = 1.0  # only used by init_mode=sqrt

    # Execution
    epochs: int = 100
    seed: int = 0
    device: str = "cuda"
    chunk_size: int = 16
    log_every: int = 1024
    eval_every: int = 1
    test_batch_size: int = 1000
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    tf32: bool = False

    data_dir: str = "./data"
    output_dir: str = "./results/stuck2025_signed_symetric"

    @property
    def n_steps(self) -> int:
        return int(round(self.presentation_ms / self.dt_ms))

    @property
    def presentation_s(self) -> float:
        return self.presentation_ms / 1000.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Data: preprocess once and keep on GPU
# =============================================================================

@torch.no_grad()
def load_gpu_dataset(cfg: Config, device: torch.device):
    train_ds = datasets.MNIST(cfg.data_dir, train=True, download=True)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True)

    raw_mean = float(train_ds.data.float().mean().item() / 255.0)
    if cfg.image_mean_target is None:
        cfg.image_mean_target = raw_mean

    ntr = len(train_ds) if cfg.train_samples is None else min(cfg.train_samples, len(train_ds))
    nte = len(test_ds) if cfg.test_samples is None else min(cfg.test_samples, len(test_ds))

    def prep(img_u8: Tensor, labels: Tensor, n: int):
        x = img_u8[:n].to(device=device, dtype=torch.float32).reshape(n, 784)
        x.div_(255.0)

        # Paper: normalize each image to a common mean.
        per_image_mean = x.mean(dim=1, keepdim=True).clamp_min_(1e-12)
        x.mul_(float(cfg.image_mean_target) / per_image_mean)
        x.clamp_(0.0, 1.0)

        rates = cfg.input_min_hz + (cfg.input_max_hz - cfg.input_min_hz) * x
        p = rates * (cfg.dt_ms / 1000.0)
        p.clamp_(0.0, 1.0)

        y = labels[:n].to(device=device, dtype=torch.long)
        return p.contiguous(), y.contiguous()

    p_train, y_train = prep(train_ds.data, train_ds.targets, ntr)
    p_test, y_test = prep(test_ds.data, test_ds.targets, nte)

    return p_train, y_train, p_test, y_test, raw_mean


# =============================================================================
# Initialization
# =============================================================================

def fluctuation_sigma(
    sigma_u: float,
    n_pre: int,
    p_event: float,
    dt_ms: float,
    tau_ms: float,
) -> float:
    """
    Discrete-time fluctuation initialization for:
        V[t+1] = a V[t] + input[t]
        a = 1 - dt/tau

    Var[V] = n*sigma_w^2*p(1-p)/(1-a^2)
    """
    a = 1.0 - dt_ms / tau_ms
    p = min(max(float(p_event), 1e-8), 1.0 - 1e-8)

    return float(
        sigma_u
        * math.sqrt((1.0 - a * a) / (float(n_pre) * p * (1.0 - p)))
    )


@torch.no_grad()
def initialize_weights(
    cfg: Config,
    device: torch.device,
    mean_input_event_probability: float,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:

    if cfg.init_mode == "fluctuation":
        sigma_w1 = fluctuation_sigma(
            sigma_u=cfg.sigma_u_target,
            n_pre=cfg.n_input,
            p_event=mean_input_event_probability,
            dt_ms=cfg.dt_ms,
            tau_ms=cfg.tau_s_ms,
        )

        # Hidden target is the best paper-specified prior for initial hidden rate.
        p_hidden = cfg.hidden_target_hz * cfg.dt_ms / 1000.0

        sigma_w2 = fluctuation_sigma(
            sigma_u=cfg.sigma_u_target,
            n_pre=cfg.n_hidden,
            p_event=p_hidden,
            dt_ms=cfg.dt_ms,
            tau_ms=cfg.tau_s_ms,
        )

    elif cfg.init_mode == "sqrt":
        sigma_w1 = cfg.init_gain / math.sqrt(cfg.n_input)
        sigma_w2 = cfg.init_gain / math.sqrt(cfg.n_hidden)

    else:
        raise ValueError(f"Unknown init_mode={cfg.init_mode!r}")

    W1 = torch.randn(
        cfg.n_hidden, cfg.n_input,
        device=device, dtype=torch.float32
    ) * sigma_w1

    W2 = torch.randn(
        cfg.n_output, cfg.n_hidden,
        device=device, dtype=torch.float32
    ) * sigma_w2

    info = {
        "sigma_w1": sigma_w1,
        "sigma_w2": sigma_w2,
        "mean_input_event_probability": mean_input_event_probability,
        "assumed_hidden_event_probability": (
            cfg.hidden_target_hz * cfg.dt_ms / 1000.0
        ),
    }
    return W1, W2, info


# =============================================================================
# Exact signed-Burstprop single-sample dynamics
# =============================================================================

def make_train_one(cfg: Config):

    a_s = cfg.dt_ms / cfg.tau_s_ms
    a_d = cfg.dt_ms / cfg.tau_d_ms
    a_e = cfg.dt_ms / cfg.tau_elig_ms
    a_r = cfg.dt_ms / cfg.tau_rate_ms
    rate_jump_hz = 1000.0 / cfg.tau_rate_ms

    @torch.no_grad()
    def train_one(
        W1: Tensor,
        W2: Tensor,
        p_in: Tensor,
        label: Tensor,
    ) -> Tensor:
        """
        Returns CUDA tensor:
        [correct,
         hidden_events, output_events,
         hidden_Bplus, hidden_Bminus,
         output_Bplus, output_Bminus,
         teacher_MSE_sum]
        """

        dev = W1.device
        dtype = W1.dtype

        vs_h = torch.zeros(cfg.n_hidden, device=dev, dtype=dtype)
        vd_h = torch.zeros_like(vs_h)

        vs_o = torch.zeros(cfg.n_output, device=dev, dtype=dtype)
        vd_o = torch.zeros_like(vs_o)

        e_in = torch.zeros(cfg.n_input, device=dev, dtype=dtype)
        e_h = torch.zeros(cfg.n_hidden, device=dev, dtype=dtype)

        rate_h = torch.zeros(cfg.n_hidden, device=dev, dtype=dtype)
        rate_o = torch.zeros(cfg.n_output, device=dev, dtype=dtype)

        out_counts = torch.zeros(cfg.n_output, device=dev, dtype=dtype)
        prev_signed_o = torch.zeros(cfg.n_output, device=dev, dtype=dtype)

        # CUDA-Graph-safe target construction.
        # Avoid: target_o[label] = target_high, because `label` is a CUDA tensor
        # and dynamic indexed writes are not permitted during stream capture.
        class_ids = torch.arange(
            cfg.n_output,
            device=dev,
            dtype=torch.long,
        )
        target_o = (
            torch.full(
                (cfg.n_output,),
                cfg.target_low_hz,
                device=dev,
                dtype=dtype,
            )
            + (class_ids == label).to(dtype)
            * (cfg.target_high_hz - cfg.target_low_hz)
        )

        # Scalar metric accumulators kept entirely on GPU.
        correct = torch.zeros((), device=dev, dtype=dtype)
        hidden_events = torch.zeros((), device=dev, dtype=dtype)
        output_events = torch.zeros((), device=dev, dtype=dtype)
        hidden_bplus = torch.zeros((), device=dev, dtype=dtype)
        hidden_bminus = torch.zeros((), device=dev, dtype=dtype)
        output_bplus = torch.zeros((), device=dev, dtype=dtype)
        output_bminus = torch.zeros((), device=dev, dtype=dtype)
        teacher_mse_sum = torch.zeros((), device=dev, dtype=dtype)

        for _ in range(cfg.n_steps):

            # ---------------------------------------------------------------
            # Input Poisson events
            # ---------------------------------------------------------------
            s_in = (torch.rand_like(p_in) < p_in).to(dtype)

            # ---------------------------------------------------------------
            # Soma: same-timestep feedforward propagation
            # ---------------------------------------------------------------
            vs_h = vs_h + a_s * (-vs_h) + torch.mv(W1, s_in)
            s_h = (vs_h >= cfg.v_threshold).to(dtype)
            vs_h = torch.where(
                s_h.bool(),
                torch.full_like(vs_h, cfg.v_reset),
                vs_h,
            )

            vs_o = vs_o + a_s * (-vs_o) + torch.mv(W2, s_h)
            s_o = (vs_o >= cfg.v_threshold).to(dtype)
            vs_o = torch.where(
                s_o.bool(),
                torch.full_like(vs_o, cfg.v_reset),
                vs_o,
            )

            out_counts.add_(s_o)

            # ---------------------------------------------------------------
            # Eligibility and rate traces
            # ---------------------------------------------------------------
            e_in = e_in + a_e * (-e_in) + s_in
            e_h = e_h + a_e * (-e_h) + s_h

            rate_h = rate_h + a_r * (-rate_h) + rate_jump_hz * s_h
            rate_o = rate_o + a_r * (-rate_o) + rate_jump_hz * s_o

            # ---------------------------------------------------------------
            # Dendritic signals
            # ---------------------------------------------------------------
            teacher_o = (
                -2.0 * cfg.gamma_teacher
                * (target_o - rate_o)
            )

            reg_h = (
                -2.0 * cfg.gamma_hidden
                * (cfg.hidden_target_hz - rate_h)
            )

            # Symmetric feedback: W2^T applied to previous-ms signed bursts.
            fb_h = torch.matmul(prev_signed_o, W2)

            vd_h = vd_h + a_d * (-vd_h + reg_h) + fb_h
            vd_o = vd_o + a_d * (-vd_o + teacher_o)

            # ---------------------------------------------------------------
            # SIGNED bursts
            # P(burst|event)=tanh(|Vd|)
            # sign(Vd) chooses B+ or B-
            # ---------------------------------------------------------------
            p_b_h = torch.tanh(torch.abs(vd_h))
            p_b_o = torch.tanh(torch.abs(vd_o))

            burst_h = (
                s_h
                * (torch.rand_like(vd_h) < p_b_h).to(dtype)
            )
            burst_o = (
                s_o
                * (torch.rand_like(vd_o) < p_b_o).to(dtype)
            )

            signed_h = burst_h * torch.sign(vd_h)
            signed_o = burst_o * torch.sign(vd_o)

            bpos_h = torch.relu(signed_h)
            bneg_h = torch.relu(-signed_h)

            bpos_o = torch.relu(signed_o)
            bneg_o = torch.relu(-signed_o)

            # Signed burst resets dendrite to baseline.
            vd_h = torch.where(
                burst_h.bool(),
                torch.zeros_like(vd_h),
                vd_h,
            )
            vd_o = torch.where(
                burst_o.bool(),
                torch.zeros_like(vd_o),
                vd_o,
            )

            # ---------------------------------------------------------------
            # Signed Burstprop plasticity:
            # dW = -eta E_pre (B+ - B-)
            # ---------------------------------------------------------------
            W1.add_(
                signed_h[:, None] * e_in[None, :],
                alpha=-cfg.eta,
            )
            W2.add_(
                signed_o[:, None] * e_h[None, :],
                alpha=-cfg.eta,
            )

            prev_signed_o = signed_o

            # All metrics remain on GPU; no CPU synchronization.
            hidden_events.add_(s_h.sum())
            output_events.add_(s_o.sum())
            hidden_bplus.add_(bpos_h.sum())
            hidden_bminus.add_(bneg_h.sum())
            output_bplus.add_(bpos_o.sum())
            output_bminus.add_(bneg_o.sum())
            teacher_mse_sum.add_(torch.mean((rate_o - target_o) ** 2))

        correct = (torch.argmax(out_counts) == label).to(dtype)

        return torch.stack(
            (
                correct,
                hidden_events,
                output_events,
                hidden_bplus,
                hidden_bminus,
                output_bplus,
                output_bminus,
                teacher_mse_sum,
            )
        )

    return train_one


def make_train_chunk(cfg: Config, train_one):
    """
    IMPORTANT:
    This is NOT a minibatch update.

    Calls train_one sequentially K times while mutating the SAME W1/W2.
    Therefore every sample sees weights modified by all earlier samples.
    """

    @torch.no_grad()
    def train_chunk(
        W1: Tensor,
        W2: Tensor,
        p_chunk: Tensor,
        y_chunk: Tensor,
    ) -> Tensor:

        total = torch.zeros(8, device=W1.device, dtype=W1.dtype)

        # This Python loop executes only ONCE during CUDA graph capture.
        # Replays execute the captured CUDA dependency graph directly.
        for k in range(cfg.chunk_size):
            total.add_(
                train_one(
                    W1,
                    W2,
                    p_chunk[k],
                    y_chunk[k],
                )
            )

        return total

    return train_chunk


# =============================================================================
# CUDA Graph capture
# =============================================================================

@torch.no_grad()
def capture_train_graph(
    cfg: Config,
    W1: Tensor,
    W2: Tensor,
    train_chunk,
    example_p: Tensor,
    example_y: Tensor,
):
    """
    Capture K sequential online samples into one CUDA graph.

    W1/W2 addresses must remain unchanged for the lifetime of the graph.
    We temporarily mutate them during warm-up/capture, then restore their values.
    """

    device = W1.device
    assert device.type == "cuda"

    K = cfg.chunk_size

    static_p = torch.empty(
        (K, cfg.n_input),
        device=device,
        dtype=torch.float32,
    )
    static_y = torch.empty(
        (K,),
        device=device,
        dtype=torch.long,
    )

    static_p.copy_(example_p[:K])
    static_y.copy_(example_y[:K])

    # Preserve exact initial weights.
    W1_initial = W1.clone()
    W2_initial = W2.clone()

    # Warm-up allocations on a side stream.
    warm_stream = torch.cuda.Stream(device=device)
    warm_stream.wait_stream(torch.cuda.current_stream(device))

    with torch.cuda.stream(warm_stream):
        for _ in range(2):
            _ = train_chunk(W1, W2, static_p, static_y)

    torch.cuda.current_stream(device).wait_stream(warm_stream)
    torch.cuda.synchronize(device)

    # Restore before actual capture.
    W1.copy_(W1_initial)
    W2.copy_(W2_initial)

    graph = torch.cuda.CUDAGraph()

    # Capture all K*100 sequential steps.
    with torch.cuda.graph(graph):
        static_stats = train_chunk(
            W1,
            W2,
            static_p,
            static_y,
        )

    torch.cuda.synchronize(device)

    # Capture itself executed once; restore again so training starts from
    # the intended initialization.
    W1.copy_(W1_initial)
    W2.copy_(W2_initial)
    torch.cuda.synchronize(device)

    del W1_initial, W2_initial

    return graph, static_p, static_y, static_stats


# =============================================================================
# Test
# =============================================================================

def make_test_batch(cfg: Config):

    a_s = cfg.dt_ms / cfg.tau_s_ms

    @torch.no_grad()
    def test_batch(
        W1: Tensor,
        W2: Tensor,
        p_in: Tensor,
        labels: Tensor,
    ) -> Tensor:

        b = p_in.shape[0]

        vs_h = torch.zeros(
            b, cfg.n_hidden,
            device=W1.device,
            dtype=W1.dtype,
        )
        vs_o = torch.zeros(
            b, cfg.n_output,
            device=W1.device,
            dtype=W1.dtype,
        )
        counts = torch.zeros_like(vs_o)

        for _ in range(cfg.n_steps):
            s_in = (torch.rand_like(p_in) < p_in).to(W1.dtype)

            vs_h = (
                vs_h
                + a_s * (-vs_h)
                + torch.matmul(s_in, W1.t())
            )
            s_h = (vs_h >= cfg.v_threshold).to(W1.dtype)
            vs_h = torch.where(
                s_h.bool(),
                torch.full_like(vs_h, cfg.v_reset),
                vs_h,
            )

            vs_o = (
                vs_o
                + a_s * (-vs_o)
                + torch.matmul(s_h, W2.t())
            )
            s_o = (vs_o >= cfg.v_threshold).to(W1.dtype)
            vs_o = torch.where(
                s_o.bool(),
                torch.full_like(vs_o, cfg.v_reset),
                vs_o,
            )
            counts.add_(s_o)

        pred = counts.argmax(dim=1)
        return (pred == labels).sum().to(torch.float32)

    return test_batch


@torch.no_grad()
def evaluate(
    cfg: Config,
    W1: Tensor,
    W2: Tensor,
    p_test: Tensor,
    y_test: Tensor,
    test_batch,
) -> float:

    correct = torch.zeros((), device=W1.device)
    n = p_test.shape[0]

    for start in range(0, n, cfg.test_batch_size):
        stop = min(start + cfg.test_batch_size, n)
        correct.add_(
            test_batch(
                W1,
                W2,
                p_test[start:stop],
                y_test[start:stop],
            )
        )

    return float(correct.cpu().item() / max(n, 1))


# =============================================================================
# Logging
# =============================================================================

def summarize(stats_cpu: Tensor, n_samples: int, cfg: Config) -> Dict[str, float]:
    s = stats_cpu.tolist()

    h_events = s[1]
    o_events = s[2]

    return {
        "acc": s[0] / max(n_samples, 1),

        "hidden_rate_hz":
            h_events
            / (max(n_samples, 1) * cfg.n_hidden * cfg.presentation_s),

        "output_rate_hz":
            o_events
            / (max(n_samples, 1) * cfg.n_output * cfg.presentation_s),

        "hidden_bplus_fraction": s[3] / max(h_events, 1.0),
        "hidden_bminus_fraction": s[4] / max(h_events, 1.0),

        "output_bplus_fraction": s[5] / max(o_events, 1.0),
        "output_bminus_fraction": s[6] / max(o_events, 1.0),

        "teacher_rate_mse":
            s[7] / max(n_samples * cfg.n_steps, 1),
    }


def save_checkpoint(path: Path, epoch: int, cfg: Config, W1: Tensor, W2: Tensor, row: dict):
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(cfg),
            "W1": W1.detach().cpu(),
            "W2": W2.detach().cpu(),
            "metrics": row,
        },
        path,
    )


# =============================================================================
# CLI / main
# =============================================================================

def parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--data-dir", default="./data")
    p.add_argument(
        "--output-dir",
        default="./results/stuck2025_signed_symetric",
    )

    p.add_argument("--train-samples", type=int, default=None)
    p.add_argument("--test-samples", type=int, default=None)

    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--log-every", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--test-batch-size", type=int, default=1000)

    p.add_argument(
        "--init-mode",
        choices=["fluctuation", "sqrt"],
        default="fluctuation",
    )
    p.add_argument(
        "--sigma-u-target",
        type=float,
        default=0.5,
        help="Target membrane std for fluctuation-driven init.",
    )
    p.add_argument(
        "--init-gain",
        type=float,
        default=1.0,
        help="Only used with --init-mode sqrt.",
    )

    p.add_argument("--image-mean-target", type=float, default=None)
    p.add_argument("--tf32", action="store_true")

    p.add_argument(
        "--smoke",
        action="store_true",
        help="256 train / 1000 test / 1 epoch.",
    )

    return p


def main():
    args = parser().parse_args()

    cfg = Config(
        device=args.device,
        epochs=args.epochs,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        chunk_size=args.chunk_size,
        log_every=args.log_every,
        eval_every=args.eval_every,
        test_batch_size=args.test_batch_size,
        init_mode=args.init_mode,
        sigma_u_target=args.sigma_u_target,
        init_gain=args.init_gain,
        image_mean_target=args.image_mean_target,
        tf32=args.tf32,
    )

    if args.smoke:
        cfg.epochs = 1
        cfg.train_samples = 256
        cfg.test_samples = 1000
        cfg.log_every = 64

    if not cfg.device.startswith("cuda"):
        raise RuntimeError(
            "This script is specifically the CUDA-Graph implementation. "
            "Use --device cuda."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch installation.")

    if cfg.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Dataset entirely on GPU
    # -------------------------------------------------------------------------
    p_train, y_train, p_test, y_test, raw_mean = load_gpu_dataset(cfg, device)

    if p_train.shape[0] < cfg.chunk_size:
        raise ValueError(
            f"Need at least chunk_size={cfg.chunk_size} training samples."
        )

    # Exact mean event probability after our normalization + clipping.
    mean_p_in = float(p_train.mean().cpu().item())

    W1, W2, init_info = initialize_weights(
        cfg,
        device,
        mean_input_event_probability=mean_p_in,
    )

    train_one = make_train_one(cfg)
    train_chunk = make_train_chunk(cfg, train_one)
    test_batch = make_test_batch(cfg)

    print("=" * 104)
    print("Stuck et al. 2025 — SIGNED Burstprop — strict online CUDA Graph")
    print("=" * 104)
    print(f"GPU                   : {torch.cuda.get_device_name(device)}")
    print(f"network               : 784 -> {cfg.n_hidden} -> 10")
    print(f"training semantics    : STRICT ONLINE, one sample updates W before next sample")
    print(f"CUDA execution chunk  : {cfg.chunk_size} sequential samples / graph replay")
    print(f"timesteps/sample      : {cfg.n_steps}")
    print(f"signed burst          : P=tanh(|Vd|), sign(Vd)->B+/B-, burst resets Vd")
    print(f"feedback              : symmetric W2^T, signed burst signal, 1-ms delay")
    print(f"eta                   : {cfg.eta:g}")
    print(f"gamma_T / gamma_R     : {cfg.gamma_teacher:g} / {cfg.gamma_hidden:g}")
    print(f"input                  : {cfg.input_min_hz:g}..{cfg.input_max_hz:g} Hz")
    print(f"raw MNIST mean        : {raw_mean:.8f}")
    print(f"common image mean     : {float(cfg.image_mean_target):.8f}")
    print(f"mean input event p/ms : {mean_p_in:.6f}")
    print(f"init mode             : {cfg.init_mode}")
    print(f"sigma_U target        : {cfg.sigma_u_target:g}")
    print(f"sigma(W1)             : {init_info['sigma_w1']:.6f}")
    print(f"sigma(W2)             : {init_info['sigma_w2']:.6f}")
    print(f"TF32                  : {cfg.tf32}")
    print("-")
    print("No autograd / optimizer / BPTT / surrogate gradients / learning minibatches.")
    print("=" * 104)

    # Save resolved config and initialization.
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "initialization": init_info,
            },
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Capture graph on the ACTUAL W tensor addresses.
    # Values are restored after capture.
    # -------------------------------------------------------------------------
    print("Capturing CUDA graph...")
    tcap = time.perf_counter()

    try:
        graph, static_p, static_y, static_stats = capture_train_graph(
            cfg,
            W1,
            W2,
            train_chunk,
            p_train[:cfg.chunk_size],
            y_train[:cfg.chunk_size],
        )
    except RuntimeError as exc:
        print()
        print("CUDA Graph capture failed.")
        print("The process should be restarted before retrying CUDA capture.")
        print(f"{type(exc).__name__}: {exc}")
        raise

    print(f"CUDA graph captured in {time.perf_counter() - tcap:.1f} s.")

    # The graph captured RNG state. We intentionally do not claim bit-identical
    # equivalence to an eager run; distributional semantics are unchanged.

    history = []
    best_acc = -1.0

    n_train = p_train.shape[0]
    K = cfg.chunk_size
    n_full = (n_train // K) * K
    tail = n_train - n_full

    global_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):

        epoch_start = time.perf_counter()

        # Shuffle once on GPU.
        order = torch.randperm(n_train, device=device)

        # One coalesced gather per epoch.
        p_epoch = p_train.index_select(0, order)
        y_epoch = y_train.index_select(0, order)

        cumulative = torch.zeros(8, device=device)
        recent = torch.zeros(8, device=device)
        recent_n = 0

        print(f"\nEpoch {epoch:03d}/{cfg.epochs:03d}")
        print("-" * 104)

        for start in range(0, n_full, K):

            # Copy the next sequential execution chunk into graph-static buffers.
            static_p.copy_(p_epoch[start:start + K])
            static_y.copy_(y_epoch[start:start + K])

            # ONE graph replay executes K samples sequentially.
            graph.replay()

            cumulative.add_(static_stats)
            recent.add_(static_stats)
            recent_n += K

            done = start + K

            if (
                recent_n >= cfg.log_every
                or done == n_full
            ):
                # Only here do we synchronize GPU -> CPU.
                recent_cpu = recent.cpu()
                cumulative_cpu = cumulative.cpu()

                mr = summarize(recent_cpu, recent_n, cfg)
                ma = summarize(cumulative_cpu, done, cfg)

                elapsed = time.perf_counter() - epoch_start
                speed = done / max(elapsed, 1e-9)
                eta_s = (n_train - done) / max(speed, 1e-9)

                print(
                    f"{done:6d}/{n_train} | "
                    f"{speed:8.1f} img/s | "
                    f"ETA {eta_s/60:6.1f} min | "
                    f"accR={mr['acc']:.3f} | "
                    f"acc={ma['acc']:.3f} | "
                    f"rH={ma['hidden_rate_hz']:.1f}Hz | "
                    f"rO={ma['output_rate_hz']:.1f}Hz | "
                    f"bH+={ma['hidden_bplus_fraction']:.3f} | "
                    f"bH-={ma['hidden_bminus_fraction']:.3f} | "
                    f"bO+={ma['output_bplus_fraction']:.3f} | "
                    f"bO-={ma['output_bminus_fraction']:.3f}"
                )

                recent.zero_()
                recent_n = 0

        # Rare tail if user chooses a dataset size not divisible by K.
        if tail:
            for i in range(n_full, n_train):
                cumulative.add_(
                    train_one(
                        W1,
                        W2,
                        p_epoch[i],
                        y_epoch[i],
                    )
                )

        train_cpu = cumulative.cpu()
        train_metrics = summarize(train_cpu, n_train, cfg)

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            ttest = time.perf_counter()
            test_acc = evaluate(
                cfg,
                W1,
                W2,
                p_test,
                y_test,
                test_batch,
            )
            test_seconds = time.perf_counter() - ttest
        else:
            test_acc = float("nan")
            test_seconds = 0.0

        epoch_seconds = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_acc": train_metrics["acc"],
            "test_acc": test_acc,
            "hidden_rate_hz": train_metrics["hidden_rate_hz"],
            "output_rate_hz": train_metrics["output_rate_hz"],
            "hidden_bplus_fraction": train_metrics["hidden_bplus_fraction"],
            "hidden_bminus_fraction": train_metrics["hidden_bminus_fraction"],
            "output_bplus_fraction": train_metrics["output_bplus_fraction"],
            "output_bminus_fraction": train_metrics["output_bminus_fraction"],
            "teacher_rate_mse": train_metrics["teacher_rate_mse"],
            "epoch_seconds": epoch_seconds,
            "test_seconds": test_seconds,
        }
        history.append(row)

        print(
            f"\nepoch={epoch:03d} "
            f"train={row['train_acc']:.4f} "
            f"test={row['test_acc']:.4f} "
            f"rH={row['hidden_rate_hz']:.2f}Hz "
            f"rO={row['output_rate_hz']:.2f}Hz "
            f"time={epoch_seconds/60:.2f}min"
        )

        with open(outdir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(
            outdir / "last.pt",
            epoch,
            cfg,
            W1,
            W2,
            row,
        )

        if not math.isnan(test_acc) and test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                outdir / "best.pt",
                epoch,
                cfg,
                W1,
                W2,
                row,
            )

        del p_epoch, y_epoch

    total = time.perf_counter() - global_start

    print("\n" + "=" * 104)
    print(f"Finished in {total/3600:.2f} h")
    print(f"Best test accuracy: {best_acc:.4f}")
    print("Paper signed+symmetric 1x100 reference: 0.972 ± 0.001")
    print(f"Results: {outdir.resolve()}")
    print("=" * 104)


if __name__ == "__main__":
    main()
