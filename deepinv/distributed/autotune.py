r"""Choose the tiling and the GPU split of :func:`deepinv.distributed.distribute` from quick probes on one GPU."""

from __future__ import annotations

import gc
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch

from deepinv.models.base import Denoiser
from deepinv.distributed.framework import DistributedContext, DistributedStackedPhysics
from deepinv.distributed.framework import distributed_data_fidelity, distributed_physics
from deepinv.distributed.strategies.utils import tiling_splitting_strategy
from deepinv.utils.tensorlist import TensorList


@dataclass
class TilingConfig:
    r"""
    Configuration returned by :class:`deepinv.distributed.AutoTuner`.

    Pass `tiling_kwargs()` to :func:`deepinv.distributed.distribute` and `inner_world_size` to
    :class:`deepinv.distributed.DistributedContext`.

    :param int inner_world_size: GPUs per image.
    :param int samples_per_step: images per step, i.e. DDP replicas (the batch size). 1 for a single image.
    :param tuple[int, ...] patch_size: tile size on each axis of `tiling_dims`.
    :param int overlap: tile overlap.
    :param int max_batch_size: maximum number of patches to process in a single batch.
    :param str checkpoint_batches: `"never"` or `"always"`.
    :param tuple[int, ...] tiling_dims: cut axes; the other spatial axes are kept whole.
    :param float peak_mb: predicted peak of torch allocated memory per GPU, in MiB. The real usage is
        higher by the allocator waste and memory outside torch, both covered by `memory_fraction`.
    :param float step_s: predicted time of one step in seconds, without communication between GPUs.
    :param list[TilingConfig] alternatives: other candidates, fastest first.
    """

    inner_world_size: int
    samples_per_step: int
    patch_size: tuple[int, ...]
    overlap: int
    max_batch_size: int
    checkpoint_batches: str
    tiling_dims: tuple[int, ...]
    peak_mb: float
    step_s: float
    alternatives: list[TilingConfig] = field(default_factory=list)

    @property
    def gpus_used(self) -> int:
        r"""Total number of GPUs, the idle ones excluded."""
        return self.inner_world_size * self.samples_per_step

    @property
    def images_per_s(self) -> float:
        r"""Predicted throughput of the whole job."""
        return self.samples_per_step / self.step_s

    def tiling_kwargs(self) -> dict:
        r"""
        Tiling arguments of :func:`deepinv.distributed.distribute`.

        :return: `patch_size`, `overlap`, `max_batch_size`, `checkpoint_batches` and `tiling_dims`.
        """
        return dict(
            patch_size=self.patch_size,
            overlap=self.overlap,
            max_batch_size=self.max_batch_size,
            checkpoint_batches=self.checkpoint_batches,
            tiling_dims=self.tiling_dims,
        )


@dataclass
class _Run:
    """Physics probe at one inner world size. Allocated bytes, weights and user tensors included."""

    m_phys: int  # peak of the step, inside the physics
    at_call: int  # memory alive while the denoiser runs, max over calls
    n_calls: int  # denoiser calls per step
    t_phys: float  # step time with an identity denoiser


@dataclass
class _Tile:
    """Tile probe for one patch size: one window through the real denoiser, batch 1.

    Memory is measured above the memory before the probe.
    """

    patch: tuple[int, ...]
    dims: tuple[int, ...]  # cut axes
    n: int  # tiles in the image
    vpad: int  # bytes of the padded image
    win: int  # bytes of one window
    peak_inf: int  # peak of a no-grad forward
    held: int  # held after a forward with graph: saved activations + output
    peak_train: int  # peak of a forward + backward
    t_fwd: float
    t_bwd: float


class AutoTuner:
    r"""
    Choose the tiling and the GPU split of :func:`deepinv.distributed.distribute` from quick probes on one GPU.

    - :meth:`min_gpus`: smallest number of GPUs per image that fits in memory.
    - :meth:`best_single`: fastest tiling of one image on a given number of GPUs.
    - :meth:`best_multi`: fastest split of the GPUs into DDP groups, one image per group.

    Everything runs on a single GPU. The physics is measured by running `step` with the denoiser
    replaced by the identity, and the denoiser by running one tile of each candidate size. The memory
    and time of each configuration are then computed from these measurements.

    .. note::

        Probe on the same GPU type and software as the target: cuDNN choices change memory by up to
        20 percent between cards. Also use the same `PYTORCH_CUDA_ALLOC_CONF` in the probe and the job: the
        default `memory_fraction` depends on it.

    .. warning::

        Communication between GPUs cannot be measured on one GPU, so predicted times leave it out.

    |sep|

    :Example:

    >>> tuner = AutoTuner(model, make_physics, step, overlap=32, optimizer=opt)  # doctest: +SKIP
    >>> cfg = tuner.best_multi(num_gpus=22)  # doctest: +SKIP
    >>> ctx = DistributedContext(inner_world_size=cfg.inner_world_size)  # doctest: +SKIP
    >>> model = distribute(model, ctx, **cfg.tiling_kwargs())  # doctest: +SKIP

    :param torch.nn.Module model: a :class:`deepinv.models.Denoiser`, or a model containing one
        (e.g. an unfolded :class:`deepinv.optim.BaseOptim`), on the GPU.
    :param Callable make_physics: `make_physics(ctx)` builds the physics of the rank of `ctx`. A
        pre-sharded physics is accepted; its partition is checked by the job, not by the probe.
    :param Callable step: `step(model, physics)`: one step on one image, with backward and
        `optimizer.step()` in training. It can wrap a :class:`deepinv.Trainer`, e.g.
        `trainer.compute_loss(physics, x, y, train=True, step=True)`, so that the probe measures the
        code the job runs. The probe plays one rank alone: a gather returns zeros for the other
        ranks' operators, which count their memory but not their values.
    :param int overlap: tile overlap.
    :param torch.optim.Optimizer optimizer: training optimizer, to count its state. `None` for inference.
    :param float gpu_memory_gb: target GPU memory. Default: the probe GPU.
    :param list[int] patch_sizes: candidate patch sizes, the same on each cut axis. Default: from the image size.
    :param float memory_fraction: usable part of the memory; the rest covers allocator waste.
        Default: 0.9 with `expandable_segments`, else 0.8.
    :param bool physics_scales: whether the physics is split over the GPUs of a group, so that its
        memory and time depend on the group size. `False` probes it once and reuses that run.
    :param tuple[int, ...] measurement_shape: shape of the whole measurement `y`, batch included.
        Gives the shape of the other ranks' results without a probe on one GPU. Default: learned
        from that probe.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        make_physics: Callable[[DistributedContext], Any],
        step: Callable[[torch.nn.Module, Any], None],
        *,
        overlap: int,
        optimizer: torch.optim.Optimizer | None = None,
        gpu_memory_gb: float | None = None,
        patch_sizes: list[int] | None = None,
        memory_fraction: float | None = None,
        physics_scales: bool = True,
        measurement_shape: tuple[int, ...] | None = None,
    ):
        self.model, self.make_physics, self.step = model, make_physics, step
        self.overlap, self.optimizer = overlap, optimizer
        self.gpu_memory_gb, self.patch_sizes = gpu_memory_gb, patch_sizes
        if memory_fraction is None:
            alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            memory_fraction = 0.9 if "expandable_segments:True" in alloc_conf else 0.8
        self.memory_fraction = memory_fraction
        self.physics_scales = physics_scales
        self.train = optimizer is not None
        self.modes = ("never", "always") if self.train else ("never",)
        self.denoiser = next(
            (m for m in model.modules() if isinstance(m, Denoiser)), None
        )
        if self.denoiser is None:
            raise ValueError("model must be or contain a deepinv.models.Denoiser")
        self._runs: dict[int, _Run | None] = {}
        self.tiles: list[_Tile] | None = None
        # measurement shape of each (operator count, operator); one operator: the whole measurement
        self._shapes: dict[tuple[int, int], torch.Size] = {}
        if measurement_shape is not None:
            self._shapes[(1, 0)] = torch.Size(measurement_shape)

    def min_gpus(self, max_gpus: int) -> tuple[int | None, int | None]:
        r"""
        Smallest number of GPUs per image that fits.

        :param int max_gpus: largest number of GPUs per image to try.
        :return: `(p_min, p_never)`: smallest inner world size that fits with `"always"` and with
            `"never"` checkpointing, `None` if none up to `max_gpus`. Same value twice in inference.
        """
        fit = lambda mode: self._first_fit(
            lambda inner: self._best(inner, (mode,)) is not None, max_gpus
        )
        return (fit("always"), fit("never")) if self.train else (fit("never"),) * 2

    def best_single(self, num_gpus: int) -> TilingConfig | None:
        r"""
        Fastest tiling for one image on `num_gpus` GPUs.

        :param int num_gpus: inner world size.
        :return: the fastest config that fits, `None` if none. Alternatives: the other patch sizes
            and modes that fit, slowest last, to fall back on if the job still OOMs.
        """
        ranked = sorted(self._configs(num_gpus), key=lambda c: c.step_s)
        if not ranked:
            return None
        best, best.alternatives = ranked[0], ranked[1:]
        return best

    def best_multi(self, num_gpus: int) -> TilingConfig | None:
        r"""
        Fastest split of `num_gpus` GPUs into DDP groups, one image per group.

        Candidates, for each checkpoint mode, from the smallest group size `p0` that fits with DDP:

        - most groups, `num_gpus // p0`, with the idle GPUs added to the groups;
        - the smallest divisor of `num_gpus` above `p0`, which uses every GPU.

        The pick has the most images per second. Within 5 percent, the one with more groups wins:
        it communicates less, which the predicted time leaves out.

        :param int num_gpus: total number of GPUs.
        :return: the pick, with `samples_per_step` groups and the other candidates as alternatives.
            `None` if nothing fits.
        """
        cands = set()
        for mode in self.modes:
            p0 = self._first_fit(
                lambda inner: self._best(inner, (mode,), ddp=True) is not None, num_gpus
            )
            if p0:
                cands.add(num_gpus // (num_gpus // p0))
                cands.add(next(d for d in range(p0, num_gpus + 1) if num_gpus % d == 0))
        ranked = [
            replace(c, samples_per_step=num_gpus // inner)
            for inner in cands
            if (c := self._best(inner, ddp=True, exact=False))
        ]
        if not ranked:
            return None
        top = max(c.images_per_s for c in ranked)
        win = max(
            (c for c in ranked if c.images_per_s >= 0.95 * top),
            key=lambda c: c.samples_per_step,
        )
        # the ranking reused an older probe: measure the winner at its own group size
        exact = self._best(win.inner_world_size, ddp=True)
        best = replace(exact, samples_per_step=win.samples_per_step) if exact else win
        best.alternatives = sorted(
            (c for c in ranked if c.inner_world_size != win.inner_world_size),
            key=lambda c: -c.images_per_s,
        )
        return best

    @staticmethod
    def _first_fit(ok: Callable[[int], bool], max_gpus: int) -> int | None:
        """
        Smallest `n` in `[1, max_gpus]` with `ok(n)`, for `ok` false then true. Doubling, then bisection.

        Monotone because memory decreases when the number of GPUs per image grows.

        :param Callable ok: test on a number of GPUs per image, usually "a config fits".
        :param int max_gpus: largest number to try.
        :return: the smallest number that passes, `None` if none does.
        """
        fail, n = 0, 1  # fail: largest n known to fail
        while not ok(n):
            if n >= max_gpus:
                return None
            fail, n = n, min(2 * n, max_gpus)
        while n - fail > 1:
            mid = (fail + n) // 2
            fail, n = (fail, mid) if ok(mid) else (mid, n)
        return n

    def _best(
        self,
        inner: int,
        modes: tuple[str, ...] | None = None,
        ddp: bool = False,
        exact: bool = True,
    ) -> TilingConfig | None:
        """Fastest config that fits at `inner`, `None` if none. See :meth:`_configs`."""
        cands = self._configs(inner, modes, ddp, exact)
        return min(cands, key=lambda c: c.step_s, default=None)

    def _configs(
        self,
        inner: int,
        modes: tuple[str, ...] | None = None,
        ddp: bool = False,
        exact: bool = True,
    ) -> list[TilingConfig]:
        """
        Configurations that fit in memory, with their predicted peak and step time.

        Peak of torch allocated memory, with `b` = `max_batch_size` and the tile memory `m0 + b * per_tile`::

            peak = extra + comm + max(m_phys, at_call + m0 + b * per_tile)
            inference:   m0 = vpad + v + 2 k win                               per_tile = peak_inf - win
            "always":    m0 = vpad + 2 v + 2 n k win + g                       per_tile = peak_train
            "never":     m0 = vpad + 2 v + 2 n k win + g + n k (held - win)    per_tile = peak_train - held

        `vpad`, `win`, `peak_inf`, `held` and `peak_train` are measured per patch size in :class:`_Tile`,
        `m_phys`, `at_call` and `n` in :class:`_Run`; `v` (one image), `g` (gradients) and `extra` in `_setup`.

        - `extra`: gradients and optimizer state. `comm`: one image if `inner > 1` (all_reduce copy),
          `g` with DDP (second copy of the gradients).
        - `m_phys` peaks inside the physics; `at_call` is what is alive while the denoiser runs. They
          never peak together, hence the max.
        - `k` tiles on the busiest rank, `n` denoiser calls. Terms of `m0`: padded image, output image
          (and its gradient in training), tile inputs and outputs (of every call in training, kept for
          backward), second copy of the gradients while they accumulate, activations of every tile.

        :param int inner: GPUs per image.
        :param tuple[int, ...] | None modes: checkpoint modes to try. Default: both, `"never"` in inference.
        :param bool ddp: whether the groups are replicas of a DDP job.
        :param bool exact: probe the physics at `inner`. `False` reuses the probe of the largest tested size
            below it, a safe over-estimate since fewer GPUs hold more operators.
        :return: one config per patch size and mode that fits, each with the largest `max_batch_size`.
        """
        size = inner if self.physics_scales else 1  # one run describes every group size
        # probes are cached; the first one also sets the budget and runs the tile probe
        if exact and size not in self._runs:
            if not self._runs:
                self._setup()
                # P=1 sees every operator: it gives the measurement shapes
                if size != 1 and (1, 0) not in self._shapes:
                    self._runs[1] = self._probe_phys(1)
            self._runs[size] = self._probe_phys(size)
            if self.tiles is None and self._runs[size]:
                self.tiles = self._probe_tiles()
        tested = [q for q, probe in self._runs.items() if probe and q <= size]
        run = self._runs[size] if exact else self._runs[max(tested)]
        if run is None:
            return []
        fixed = self.extra
        # the functional all_reduce clones its input
        fixed += self.v if inner > 1 else 0
        # DDP keeps a second copy of the gradients
        fixed += self.g if ddp and self.train else 0
        if fixed + run.m_phys > self.budget:
            return []
        room = self.budget - fixed - run.at_call
        n, v, cands = run.n_calls, self.v, []
        for tile in self.tiles:
            k = math.ceil(tile.n / inner)  # tiles on the busiest rank
            for mode in modes or self.modes:
                if not self.train:
                    m0 = tile.vpad + v + 2 * k * tile.win
                    per_tile = tile.peak_inf - tile.win
                else:  # + input grads, inputs and outputs of every call kept, second gradient copy
                    m0 = tile.vpad + 2 * v + 2 * n * k * tile.win + self.g
                    if mode == "always":  # one batch recomputed, then its backward
                        per_tile = tile.peak_train
                    else:  # activations of every tile kept
                        act = tile.held - tile.win  # saved activations of one window
                        m0 += n * k * act
                        # backward on top of the saved activations and the output
                        per_tile = tile.peak_train - tile.held
                mb = k if per_tile <= 0 else min(k, int((room - m0) // per_tile))
                if mb < 1 or m0 + mb * per_tile > room:
                    continue
                t_win = tile.t_fwd + tile.t_bwd
                t_win += tile.t_fwd if mode == "always" else 0  # recomputed forward
                step_s = n * k * t_win + run.t_phys
                peak = fixed + max(run.m_phys, run.at_call + m0 + mb * per_tile)
                cands.append(
                    TilingConfig(
                        inner_world_size=inner,
                        samples_per_step=1,
                        patch_size=tile.patch,
                        overlap=self.overlap,
                        max_batch_size=mb,
                        checkpoint_batches=mode,
                        tiling_dims=tile.dims,
                        peak_mb=peak / 2**20,
                        step_s=step_s,
                    )
                )
        return cands

    def _setup(self):
        """Memory budget, and the bytes of gradients and optimizer state, which the probes do not hold."""
        self.device = next(self.denoiser.parameters()).device
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        if self.gpu_memory_gb:  # target memory minus this card's context
            mem = self.gpu_memory_gb * 2**30 - (total - free - reserved)
        else:
            mem = free + reserved
        self.budget = self.memory_fraction * mem
        params = []
        if self.train:
            params = [p for g in self.optimizer.param_groups for p in g["params"]]
        self.g = sum(p.numel() * p.element_size() for p in params)
        self.extra = 0
        if self.train:
            with _restored(self.model, self.optimizer):
                base = torch.cuda.memory_allocated(self.device)
                for p in params:
                    p.grad = torch.zeros_like(p)
                self.optimizer.step()  # zero gradients: creates the state, weights restored after
                self.extra = torch.cuda.memory_allocated(self.device) - base

    def _probe_phys(self, inner: int) -> _Run | None:
        """
        One `step` with the denoiser replaced by the identity, as one GPU of a group of `inner`.

        The physics operators are split over the `inner` GPUs of a group, so its memory and time depend
        on `inner`: the probe plays rank 0, which holds the most operators.

        :param int inner: GPUs per image.
        :return: the measurements of the step, `None` on out of memory.
        """
        calls = []
        dummy = torch.zeros((), device=self.device, requires_grad=True)

        def identity(x, *args, **kwargs):
            """Stand-in denoiser: records the memory alive at each call."""
            if not calls:  # shape and other arguments (sigma), reused by the tile probe
                detach = lambda a: a.detach() if torch.is_tensor(a) else a
                self._call = (
                    x.shape,
                    x.dtype,
                    [detach(a) for a in args],
                    {k: detach(a) for k, a in kwargs.items()},
                )
            calls.append(torch.cuda.memory_allocated(self.device))
            return x + dummy  # a new image with a graph, as the real denoiser returns

        # not entered: no process group, collectives return their input
        ctx = DistributedContext()
        ctx.inner_world_size, ctx.inner_rank, ctx.device = inner, 0, self.device
        with (
            _restored(self.model, self.optimizer),
            _no_shard_check(),
            _filled_gather(self._shapes, self.train),
        ):
            torch.cuda.reset_peak_memory_stats(self.device)
            self.denoiser.forward = identity  # on the instance: every reference sees it
            gc.disable()  # reference cycles would be freed at random moments
            try:
                physics = self.make_physics(ctx)
                t0 = _now()
                self.step(self.model, physics)
                t_phys = _now() - t0
                del physics
            except torch.OutOfMemoryError:
                return None
            finally:
                del self.denoiser.forward
                gc.enable()
            peak = torch.cuda.max_memory_allocated(self.device)
        if not calls:
            raise RuntimeError("step did not call the denoiser")
        return _Run(peak, max(calls), len(calls), t_phys)

    def _probe_tiles(self) -> list[_Tile]:
        """
        One window per patch candidate through the real denoiser, largest first.

        :return: the measurements of each candidate. Windows that run out of memory are skipped.
        """
        shape, dtype, args, kwargs = self._call
        elem = torch.empty((), dtype=dtype).element_size()
        self.v = math.prod(shape) * elem
        tiles = []
        for patch, dims in _patch_candidates(shape[2:], self.overlap, self.patch_sizes):
            slices, meta = tiling_splitting_strategy(
                shape, patch_size=patch, overlap=self.overlap, tiling_dims=dims
            )
            pad = meta["global_padding"]
            padded = list(shape)
            # F.pad order: last dimension first, two numbers per axis
            for i, (before, after) in enumerate(zip(pad[::2], pad[1::2], strict=True)):
                padded[-1 - i] += before + after
            vpad = elem * math.prod(padded)
            window = list(shape)
            for dim, w in zip(dims, meta["window_shape"], strict=True):
                window[dim] = w
            try:
                x = torch.randn(window, device=self.device, dtype=dtype)
                win = x.numel() * x.element_size()
                with _restored(self.model, self.optimizer):
                    # gradients exist as in training; their second copy is g
                    if self.train:
                        for p in self.denoiser.parameters():
                            p.grad = torch.zeros_like(p) if p.requires_grad else None
                    base = torch.cuda.memory_allocated(self.device)
                    with torch.no_grad():
                        self.denoiser(x, *args, **kwargs)  # warm-up
                        torch.cuda.reset_peak_memory_stats(self.device)
                        t0 = _now()
                        self.denoiser(x, *args, **kwargs)
                        t_fwd = _now() - t0
                    peak_inf = torch.cuda.max_memory_allocated(self.device) - base
                    held = peak_train = t_bwd = 0
                    if self.train:
                        # warm-up: the backward kernels are chosen on the first call
                        y = self.denoiser(x, *args, **kwargs)
                        (y**2).mean().backward()
                        del y
                        torch.cuda.reset_peak_memory_stats(self.device)
                        y = self.denoiser(x, *args, **kwargs)
                        held = torch.cuda.memory_allocated(self.device) - base
                        t0 = _now()
                        (y**2).mean().backward()
                        t_bwd = _now() - t0
                        peak_train = torch.cuda.max_memory_allocated(self.device) - base
                        del y
                tiles.append(
                    _Tile(
                        patch,
                        dims,
                        len(slices),
                        vpad,
                        win,
                        peak_inf,
                        held,
                        peak_train,
                        t_fwd,
                        t_bwd,
                    )
                )
            except torch.OutOfMemoryError:
                pass
            # y survives an out of memory in the backward: its graph must not reach the next probe
            x = y = None
        if not tiles:
            raise RuntimeError("no patch size fits on the probe GPU")
        return tiles


def _patch_candidates(
    spatial: tuple[int, ...],
    overlap: int,
    sizes: list[int] | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    (patch, cut axes) candidates, largest tiles first.

    `sizes` if given, cutting each axis longer than the patch. Otherwise 2, 3, 4, 6, 8, ... tiles on the
    longest axis: each axis is split evenly into tiles of at most that length (not cut if one tile covers
    it), windows rounded up to a multiple of 16, tiles of at least `max(32, 2 * overlap)` pixels.

    :param tuple[int, ...] spatial: spatial shape of the image.
    :param int overlap: tile overlap.
    :param list[int] sizes: patch sizes asked by the user. Default: the rule above.
    :return: `(patch, cut axes)` pairs, as :func:`deepinv.distributed.distribute` takes them.
    """
    ndim = len(spatial)
    if sizes:  # user sizes: cut every axis longer than the patch
        out = []
        for p in sorted(sizes, reverse=True):
            dims = tuple(i - ndim for i, D in enumerate(spatial) if D > p)
            if dims:
                out.append(((p,) * len(dims), dims))
        return out
    longest, min_tile = max(spatial), max(32, 2 * overlap)
    out, n_tiles, n_next = [], 2, 3
    while longest / n_tiles >= min_tile:
        patch, dims = [], []
        for i, D in enumerate(spatial):
            # tiles on this axis, none longer than longest / n_tiles
            n_axis = math.ceil(D * n_tiles / longest)
            # multiple of 16: U-Nets such as DRUNet (8) halve the image several times
            window = math.ceil((math.ceil(D / n_axis) + 2 * overlap) / 16) * 16
            if n_axis > 1 and window - 2 * overlap < D:  # the tiler needs patch < axis
                patch.append(window - 2 * overlap)
                dims.append(i - ndim)
        if (tuple(patch), tuple(dims)) not in out:
            out.append((tuple(patch), tuple(dims)))
        # 2, 3, 4, 6, 8, 12, ...: two tile counts per doubling
        n_tiles, n_next = n_next, 2 * n_tiles
    return out


def _now() -> float:
    """Clock of the GPU work queued so far, in seconds."""
    torch.cuda.synchronize()
    return time.perf_counter()


def _cpu(obj):
    """
    Copy of a nested state dict with every tensor on the CPU.

    :param Any obj: tensor, or a list, tuple or dict of them.
    :return: the same structure, with the tensors copied to the CPU.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_cpu(v) for v in obj)
    return obj


@contextmanager
def _no_shard_check():
    """
    Skip the check that a pre-sharded physics covers every operator exactly once.

    It asks every rank which operators it owns, and a probe is one rank alone. The real run, where
    the ranks exist, validates the partition.
    """
    check = getattr(DistributedStackedPhysics, "_validate_shard_ownership", None)
    if check is None:  # renamed upstream: let the physics decide
        yield
        return
    DistributedStackedPhysics._validate_shard_ownership = lambda self: None
    try:
        yield
    finally:
        DistributedStackedPhysics._validate_shard_ownership = check


@contextmanager
def _filled_gather(shapes: dict[tuple[int, int], torch.Size], train: bool):
    """
    Fill the holes of a gather with zeros, as if the other ranks had answered.

    :param dict[tuple[int, int], torch.Size] shapes: measurement shape of each
        `(operator count, operator)`, updated at each gather.
    :param bool train: whether the zeros require gradients.
    """
    original = distributed_physics.map_reduce_gather

    def gather(*args, **kwargs):
        out = original(*args, **kwargs)
        if not isinstance(out, TensorList):
            return out  # a reduction
        n, local = len(out.x), [r for r in out.x if r is not None]
        holes = [i for i, r in enumerate(out.x) if r is None]
        shapes.update({(n, i): r.shape for i, r in enumerate(out.x) if r is not None})
        if not holes or not local:
            return out
        shape = list(local[0].shape)
        whole = shapes.get((1, 0), shape)  # one operator: the whole measurement
        axes = [d for d, (a, b) in enumerate(zip(whole, shape, strict=True)) if a != b]
        if len(axes) == 1:  # the operators cut this axis
            d = axes[0]
            rest = whole[d] - sum(r.shape[d] for r in local)
        for k, i in enumerate(holes):
            if len(axes) == 1:  # the holes share the rest evenly
                shape[d] = rest // len(holes) + int(k < rest % len(holes))
            out.x[i] = torch.zeros(
                shapes.get((n, i), shape),
                dtype=local[0].dtype,
                device=local[0].device,
                requires_grad=train,
            )
        return out

    modules = (distributed_physics, distributed_data_fidelity)
    for m in modules:
        m.map_reduce_gather = gather
    try:
        yield
    finally:
        for m in modules:
            m.map_reduce_gather = original


@contextmanager
def _restored(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None):
    """
    Undo what a probe does to weights, gradients and optimizer state. The saved copy is on the CPU.

    :param torch.nn.Module model: model to restore.
    :param torch.optim.Optimizer optimizer: optimizer to restore. `None` in inference.
    """
    weights = _cpu(model.state_dict())
    grads = [(p, p.grad) for p in model.parameters()]
    state = _cpu(optimizer.state_dict()) if optimizer is not None else None
    try:
        yield
    finally:
        model.load_state_dict(weights)
        for p, g in grads:
            p.grad = g
        if state is not None:
            optimizer.load_state_dict(state)
        gc.collect()
        torch.cuda.empty_cache()
