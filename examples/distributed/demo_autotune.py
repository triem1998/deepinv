"""
Choosing the Tiling and the GPU Split with AutoTuner
====================================================

Distributing a reconstruction over several GPUs asks for four numbers: how many GPUs per image, how
large the tiles are, how many of them a GPU processes at once, and whether the tiles are recomputed
during the backward pass. Guessing them costs one out-of-memory error per guess, and a multi-GPU job
is slow to restart.

:class:`deepinv.distributed.AutoTuner` measures them instead, and it does so **on a single GPU**: the
physics is run once with the denoiser replaced by the identity, and the denoiser is run on one tile of
each candidate size. The memory and the time of every configuration are then computed from those
measurements, which answers three questions:

- :meth:`deepinv.distributed.AutoTuner.min_gpus`: the smallest number of GPUs per image that fits;
- :meth:`deepinv.distributed.AutoTuner.best_single`: the fastest tiling of one image on a given number of GPUs;
- :meth:`deepinv.distributed.AutoTuner.best_multi`: the fastest split of the GPUs into data-parallel groups.

**Usage:**

.. code-block:: bash

    python examples/distributed/demo_autotune.py

This example needs a CUDA device, and a single one: no multi-GPU run is involved.

**Key Steps:**

1. Build the problem: an image, its measurements and a denoiser
2. Write `make_physics(ctx)`, which builds the operators of one rank
3. Write `step(model, physics)`, which runs one step on one image
4. Ask :class:`deepinv.distributed.AutoTuner` the three questions
5. Pass the answer to :func:`deepinv.distributed.distribute`
6. Compare the predicted memory with the real one
"""

# %%
import torch

from deepinv.distributed import AutoTuner, DistributedContext, distribute
from deepinv.models import DRUNet
from deepinv.physics import GaussianNoise
from deepinv.physics.blur import Blur
from deepinv.physics.functional import gaussian_blur
from deepinv.utils.demo import load_example

device = torch.device("cuda")
img_size = 1024  # the image the job will reconstruct
num_operators = 4  # blurs to spread over the GPUs of a group
overlap = 32  # halo of the denoiser

x = load_example("CBSD_0010.png", img_size=img_size, device=device)
denoiser = DRUNet(pretrained="download", device=device)

# %%
# The two callbacks
# -----------------
# The tuner needs to know how to build the physics of one rank, and what one step does. It calls them
# itself, on one GPU, once per group size it wants to measure.
#
# ``make_physics(ctx)`` builds the operators of the rank described by ``ctx``. Passing a factory to
# :func:`deepinv.distributed.distribute` is the simplest way: each rank then builds only the operators
# it owns, and the tuner can emulate any group size.


def build_blur(index, device, _=None):
    """One of the operators: a Gaussian blur of its own width."""
    return Blur(
        filter=gaussian_blur(sigma=(1.0 + index / 2,) * 2, device=str(device)),
        padding="circular",
        device=device,
        noise_model=GaussianNoise(sigma=0.05),
    )


def make_physics(ctx):
    return distribute(
        build_blur, ctx, type_object="linear_physics", num_operators=num_operators
    )


# %%
# ``y`` is the measurement, as a dataset provides it. Build it outside the step: the probe plays one
# rank alone, so a gather inside the step would come back incomplete.

with torch.no_grad():
    y = [build_blur(i, device)(x) for i in range(num_operators)]

# %%
# ``step(model, physics)`` runs one step on one image. The tuner replaces the denoiser by the identity
# when it measures the physics, and runs the denoiser alone when it measures the tiles. In training,
# the step also does the backward pass and ``optimizer.step()``, and the optimizer is passed to the
# tuner so that the gradients and its state are counted.


def step(model, physics):
    with torch.no_grad():
        model(physics.A_adjoint(y), 0.05)


tuner = AutoTuner(denoiser, make_physics, step, overlap=overlap)

# %%
# A training step is written the same way, and can wrap a :class:`deepinv.Trainer`, so that the probe
# measures the very code the job runs. The optimizer is passed to the tuner, which then also counts the
# gradients and the optimizer state:
#
# .. code-block:: python
#
#     trainer = dinv.Trainer(model=model, physics=None, optimizer=optimizer, losses=..., ...)
#     trainer.setup_train(train=True)
#
#     def step(model, physics):
#         trainer.compute_loss(physics, x, y, train=True, epoch=0, step=True)
#
#     tuner = AutoTuner(model, make_physics, step, overlap=overlap, optimizer=optimizer)

# %%
# The answers
# -----------
# The first call runs the probes, a few seconds; the others reuse them.

p_min, _ = tuner.min_gpus(max_gpus=8)
print(f"smallest group that fits: {p_min} GPUs")

for num_gpus in (1, 2, 4):
    cfg = tuner.best_single(num_gpus)
    if cfg is None:
        print(f"{num_gpus} GPUs: nothing fits")
        continue
    print(
        f"{num_gpus} GPUs: patch {cfg.patch_size}, max_batch_size {cfg.max_batch_size}, "
        f"{cfg.peak_mb:.0f} MiB, {cfg.step_s:.2f} s"
    )

# %%
# Every configuration that fits is kept, fastest first, so a run that still goes out of memory has a
# fallback ready.

cfg = tuner.best_single(2)
for other in cfg.alternatives[:3]:
    print(
        f"alternative: patch {other.patch_size}, {other.peak_mb:.0f} MiB, {other.step_s:.2f} s"
    )

# %%
# With several images to reconstruct, the GPUs are split into groups, one image per group, the groups
# running as data-parallel replicas.

multi = tuner.best_multi(num_gpus=8)
print(
    f"8 GPUs: {multi.samples_per_step} groups of {multi.inner_world_size}, "
    f"{multi.images_per_s:.2f} images/s"
)

# %%
# Using the answer
# ----------------
# :meth:`deepinv.distributed.TilingConfig.tiling_kwargs` returns the tiling arguments of
# :func:`deepinv.distributed.distribute`, and ``inner_world_size`` goes to the context. The job is then
# launched with ``torchrun``; the single process below only shows that the configuration is accepted.

cfg = tuner.best_single(1)
with DistributedContext(inner_world_size=cfg.inner_world_size, seed=0) as ctx:
    physics = make_physics(ctx)
    model = distribute(denoiser, ctx, **cfg.tiling_kwargs())

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(physics.A_adjoint(y), 0.05)
    peak = torch.cuda.max_memory_allocated(device) / 2**20

print(f"predicted {cfg.peak_mb:.0f} MiB, measured {peak:.0f} MiB")

# %%
# The prediction covers the memory that torch allocates. The real usage is higher by the allocator
# waste and by whatever a library holds outside torch, which the ``memory_fraction`` margin absorbs.
# Probe on the same GPU type and with the same ``PYTORCH_CUDA_ALLOC_CONF`` as the job.
