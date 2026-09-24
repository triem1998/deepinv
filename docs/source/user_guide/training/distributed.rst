.. _distributed-training:

Distributed Training
====================

The distributed framework can be used during training when a
forward model is too large for one GPU, for example because the reconstruction
model processes large images or volumes, and the operator acts globally over them.

The API is the same as for :ref:`distributed reconstruction <distributed>`:

1. :class:`deepinv.distributed.DistributedContext` manages the processes and devices
2. :func:`deepinv.distributed.distribute` converts deepinv objects to distributed versions

DeepInv's intra-sample distribution can also be combined with `PyTorch
DistributedDataParallel <https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
(DDP). In that hierarchical configuration, small groups of ranks cooperate on
large samples and DDP processes independent samples across those groups.

.. warning::

    This module is in beta and may undergo significant changes in future releases.
    Some features are experimental and only supported for specific use cases.
    Please report any issues you encounter on our `GitHub repository <https://github.com/deepinv/deepinv>`_.


Quick Start
-----------

The typical workflow is to distribute the physics and then distribute the
trainable unfolded model:

.. code-block:: python

    import torch
    import deepinv as dinv
    from deepinv.distributed import DistributedContext, distribute
    from deepinv.optim import DRS
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.prior import PnP

    with DistributedContext(seed=0, seed_offset=False) as ctx:

        # distribute the forward operator
        physics = distribute(stacked_physics, ctx)

        model = DRS(
            data_fidelity=L2(),
            prior=PnP(denoiser),
            max_iter=5,
            unfold=True,
            trainable_params=["stepsize", "sigma_denoiser"],
        )

        # distribute the model
        model = distribute(model, ctx, patch_size=256, overlap=64)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = dinv.Trainer(
            model=model,
            physics=physics,
            optimizer=optimizer,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            device=ctx.device,
            verbose=(ctx.rank == 0),
            show_progress_bar=(ctx.rank == 0),
        )
        trainer.train()

The rest of your training code can stay close to a standard
:class:`deepinv.Trainer` workflow. The distributed objects handle the
communication needed by the forward pass and by backpropagation.

.. note::

    See :ref:`sphx_glr_auto_examples_distributed_demo_unrolled_distributed.py`
    for a complete example training an unfolded model.


When to Use It
--------------

Distributed training is useful when you want to train a model but one process
cannot hold the full computation or would be too slow. Common scenarios include:

**Many physics operators**: if your measurements can be split across several sub-operators
:math:`A_i`, the framework can split them across ranks. Each rank applies its
local operators and the results are combined automatically.

**Large images or volumes**: if your denoiser or prior is too expensive to run
on the full signal, the framework can split the signal into overlapping tiles,
process the tiles on different ranks, and blend them back together.

**Unfolded algorithms**: unfolded models such as :class:`deepinv.optim.DRS` can
be distributed in one call when they are created with ``unfold=True``. The
framework distributes their data-fidelity terms, tiled denoisers, and trainable
algorithm parameters.


Data Parallelism
----------------

Using every available GPU for one sample minimizes the latency of that sample,
but it is not generally the best way to maximize training throughput. Once an
inverse problem fits on a smaller group of GPUs, the remaining GPUs can process
independent samples concurrently. This is especially effective when each
forward/backward pass is long, because DDP gradient communication is then small
relative to the computation between synchronizations.

For :math:`N` total processes and :math:`g` processes per sample, DeepInv can run

.. math::

    R = N/g

independent replicas. If one sample takes :math:`T(g)`, the relevant benchmark is

.. math::

    Q(g) = \frac{N/g}{T(g)}.

A useful rule is to use the minimum intra-sample parallelism that makes the
problem memory-feasible and reasonably compute-efficient, then use the remaining
processes for data parallelism. Activation checkpointing remains an independent
trade-off: it reduces stored activations but does not reduce the instantaneous
memory needed by a layer's feature maps.

Configure the topology with ``inner_world_size``:

.. code-block:: python

    with DistributedContext(inner_world_size=4, seed=0) as ctx:
        ...

For 32 launched processes, this creates eight replicas with four inner ranks
per sample. Contiguous ranks form the inner groups ``[0, 1, 2, 3]``,
``[4, 5, 6, 7]``, and so on. Ranks at the same position in those groups form the
DDP groups.

The context exposes both coordinates and process groups:

.. code-block:: python

    ctx.global_rank
    ctx.global_world_size
    ctx.inner_group
    ctx.inner_rank
    ctx.inner_world_size
    ctx.dp_group
    ctx.dp_rank
    ctx.dp_world_size

DeepInv operations use ``ctx.inner_group`` automatically. Distribute the model
for one sample first, then wrap it with DDP across replicas:

.. code-block:: python

    model = distribute(model, ctx, patch_size=256, overlap=64)
    model = ctx.distributed_data_parallel(model)

This method constructs PyTorch DDP with ``process_group=ctx.dp_group``.
When inner parallelism is also active, DeepInv installs a DDP communication
hook that reduces complete gradient buckets over the global process group. This
single owner avoids races between DDP bucket finalization and DeepInv's
end-of-backward parameter synchronization. Standard DDP options can be passed
directly:

.. code-block:: python

    model = ctx.distributed_data_parallel(
        model,
        static_graph=True,
        find_unused_parameters=False,
    )

When there is only one data-parallel replica, the method returns the original
model unchanged and DeepInv continues to synchronize gradients over the inner
group. Do not replace :meth:`DistributedContext.distributed_data_parallel
<deepinv.distributed.DistributedContext.distributed_data_parallel>` with a
manual DDP wrapper in a hierarchical configuration: the manual wrapper would
omit DeepInv's global gradient-bucket hook.

The logical reduction still has two roles: combining contributions from ranks
that decompose one inverse problem, and combining independent sample gradients.
In hierarchical DDP these operations are implemented as one bucket collective.
DDP waits until all uses of a parameter have accumulated, so repeated calls to
the same denoiser in an unfolded model are reduced once per backward pass.

Gradient Reduction Semantics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient synchronization uses an explicit reduction selected on the context:

.. code-block:: python

    with DistributedContext(
        inner_world_size=4,
        gradient_reduction="mean",
    ) as ctx:
        ...

``"mean"`` is the default and keeps gradient scale independent of the number
of participating processes. ``"sum"`` preserves the unnormalized sum. The
selected meaning does not change when differentiating through the functional
DeepInv synchronization path.
Pure data parallelism (``inner_world_size=1``) uses PyTorch standard DDP
reducer, which always averages gradients. DeepInv therefore rejects
``gradient_reduction="sum"`` in that configuration instead of silently
performing a different reduction. Hierarchical and inner-only configurations
support both options.
The legacy ``average`` option for explicitly distributed parameters overrides
the context outside DDP. A DDP model requires all its parameters to agree with
``ctx.gradient_reduction``, because one bucket cannot safely mix SUM and mean
semantics.

DDP accepts an :class:`torch.nn.Module`, not a standalone
:class:`torch.nn.Parameter`. Parameters that should participate in data
parallel synchronization must be registered on the wrapped module. In
particular, :func:`deepinv.distributed.distribute` automatically registers and
synchronizes the ``params_algo`` parameters of an unfolded ``BaseOptim`` model.
An explicitly distributed parameter that remains outside the DDP-wrapped model
is synchronized only over the inner group.

DDP gradient buckets support first-order training only. Calling backward with
``create_graph=True`` on a model returned by
``ctx.distributed_data_parallel()`` raises an explicit error instead of
silently producing invalid meta-gradients. Higher-order differentiation remains
available without DDP, where DeepInv uses autograd-aware functional
collectives. This limitation does not affect repeated denoiser calls in ordinary
first-order training.


Simple Training Pattern
-----------------------

**Step 1: Use a synchronized context**

.. code-block:: python

    with DistributedContext(seed=seed, seed_offset=False) as ctx:
        ...

Using ``seed_offset=False`` keeps random streams aligned across ranks. This is
important because ranks should usually consume the same data in the same order.

**Step 2: Distribute the physics**

.. code-block:: python

    physics = distribute(stacked_physics, ctx)

For a :class:`stacked physics <deepinv.physics.StackedPhysics>` :math:`A = [A_1, \ldots, A_N]`, each rank owns a subset of
the operators. The global forward, adjoint, and data-fidelity computations are
assembled from these local contributions.

**Step 3: Distribute the unfolded model**

.. code-block:: python

    model = distribute(
        model,
        ctx,
        patch_size=256,
        overlap=64,
        max_batch_size=1,
    )

For unfolded models, :func:`deepinv.distributed.distribute` replaces the
denoiser inside compatible priors by a tiled distributed processor and adds
gradient synchronization for trainable parameters such as stepsizes.

**Step 4: Train as usual**

.. code-block:: python

    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        device=ctx.device,
        verbose=(ctx.rank == 0),
        show_progress_bar=(ctx.rank == 0),
    )
    trainer.train()

All ranks must enter the training loop. For printing, plotting, and progress
bars, it is usually best to do the visible work only on rank 0.


Data Loading
------------

Without data parallelism, the dataloader should return the same batch on every
rank because the full distributed world cooperates on one inverse problem.

In practice:

- Do not use :class:`torch.utils.data.distributed.DistributedSampler` for this use case.
- Use the same dataset, batch size, and shuffle seed on all ranks.
- Move both images and measurements to ``ctx.device``.
- If measurements are stored as a list or :class:`deepinv.utils.TensorList`, move each tensor to the device.

With hierarchical data parallelism, samples must be sharded across replicas, not
individual processes. Use the context helper:

.. code-block:: python

    sampler = ctx.distributed_data_sampler(dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
    )

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            ...

All inner ranks receive the same indices, while different ``dp_rank`` values
receive different dataset partitions. Do not construct a default
:class:`torch.utils.data.distributed.DistributedSampler` over the global world,
because that would give cooperating inner ranks different samples.

When ``inner_world_size`` is explicitly configured, ``seed_offset=True`` offsets
the context seed by ``dp_rank``. Inner ranks therefore start from the same random
state and independent replicas use different states. Matching seeds are not
enough if different inner ranks execute a different number of random operations;
for stochastic physics, crops, masks, or measurement generation, generate random
parameters on inner rank zero and broadcast them when strict agreement is needed.

How Backward Works
------------------

You normally do not need to write custom backward code. The distributed objects
are built so that PyTorch autograd can follow the full computation.

**Through distributed physics**: each rank applies its local operators
:math:`A_i`. During backward, the gradient with respect to the shared input is
computed from local contributions and synchronized across ranks. For linear
physics, adjoints are reduced so that the model sees
the gradient of the full stacked problem, not only the local operators.

**Through data fidelity terms**: losses such as :class:`deepinv.optim.data_fidelity.L2`
are evaluated locally on each rank's measurements, then reduced. Their gradients
are propagated back through the corresponding local physics operators and
combined automatically.

**Through tiled denoisers and priors**: the input is split into overlapping
patches. Each rank processes a group of patches with the same denoiser weights.
The processed patches are blended back into the full image, and backward sends
gradients through the same patch operations. Gradients of replicated trainable
weights are synchronized so that all ranks keep the same model parameters after
the optimizer step.


Checkpointing
-------------

Distributed tiled processing can use activation checkpointing to reduce memory
during training. Instead of storing every intermediate activation for every
patch batch, PyTorch recomputes some patch forwards during backward.

The default setting is usually enough:

.. code-block:: python

    model = distribute(model, ctx, checkpoint_batches="auto")

The available modes are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Description
   * - ``"auto"``
     - Checkpoint patch batches only when gradients are enabled and there are multiple local batches.
   * - ``"always"``
     - Checkpoint patch batches whenever gradients are enabled.
   * - ``"never"``
     - Disable checkpointing.

Set ``max_batch_size=1`` to process local patches sequentially when memory is
tight. This is slower, but often allows training on larger images or volumes.


Choosing the Tiling
-------------------

:class:`deepinv.distributed.AutoTuner` picks ``inner_world_size``, ``patch_size``,
``max_batch_size`` and ``checkpoint_batches`` from quick probes on one GPU: one step with
the denoiser replaced by an identity per tested ``inner_world_size``, and one window
per patch size. No training and no multi-GPU run.

.. code-block:: python

    make_physics = lambda ctx: distribute(factory, ctx, num_operators=ctx.inner_world_size)

    def step(model, physics):
        loss = mse(model(physics(x), physics), x)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

    tuner = AutoTuner(model, make_physics, step, overlap=32, optimizer=optimizer, gpu_memory_gb=32)
    tuner.min_gpus(max_gpus=64)       # smallest group size, with "always" and with "never"
    tuner.best_single(num_gpus=8)     # one image on 8 GPUs
    cfg = tuner.best_multi(num_gpus=22)  # groups x GPUs per group, for the most images/s

    ctx = DistributedContext(inner_world_size=cfg.inner_world_size)
    model = distribute(model, ctx, **cfg.tiling_kwargs())

Probe on the same GPU type and software as the target, and use the same
``PYTORCH_CUDA_ALLOC_CONF`` in the probe and the job: the default ``memory_fraction`` depends on it.

.. note::

    See :ref:`sphx_glr_auto_examples_distributed_demo_autotune.py` for a complete example.

Pass ``physics_scales=False`` when every rank builds the full operator (``num_operators=None``) or
when the measurements are replicated: the physics is then probed once instead of once per group size.


Running Multi-Process
---------------------

Use ``torchrun`` to launch one process per GPU:

.. code-block:: bash

    torchrun --nproc_per_node=4 my_training_script.py

For example, four processes with two processes per sample create two DDP
replicas:

.. code-block:: bash

    torchrun --standalone --nproc_per_node=4 \
        examples/distributed/demo_hierarchical_training.py --inner-world-size 2

See :ref:`sphx_glr_auto_examples_distributed_demo_hierarchical_training.py` for
the complete example.

The same script also works in single-process mode:

.. code-block:: bash

    python my_training_script.py

:class:`deepinv.distributed.DistributedContext` detects whether distributed
environment variables are present and selects the device for each rank.


Troubleshooting
---------------

**Training hangs**

- Make sure every rank enters the same training loop.
- Avoid branching around forward or backward calls unless every rank follows the same branch.

**Gradients differ across ranks**

- Use ``seed_offset=False`` when all ranks should consume the same random operations.
- Make sure the same minibatch is loaded on every rank.
- Do not use a data-parallel sampler unless you intentionally changed the training strategy.

**Out of memory errors**

- Reduce ``patch_size`` and ``overlap`` to reduce the patch memory footprint.
- Set ``max_batch_size=1`` to process fewer patches at once.
- Use ``checkpoint_batches="always"`` for the distributed denoiser or unfolded model.


See Also
--------

- **Complete example**: :ref:`sphx_glr_auto_examples_distributed_demo_unrolled_distributed.py`
- **Distributed reconstruction guide**: :ref:`distributed`
- **Trainer guide**: :ref:`trainer`
- **Multi-GPU training guide**: :ref:`multigpu`
- **API Reference**: :doc:`/api/deepinv.distributed`
