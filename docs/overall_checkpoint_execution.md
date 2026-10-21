# Sequential checkpoint attempt execution

The development executor in `analysis/checkpoint_execution/` connects a sampler
and archive worker to a separate diagnostic worker. The sampling interpreter can
release its GPU context and compiled executables before the CPU environment
loads the bounded diagnostic implementation. The GPU environment need not import
the diagnostic package. Both children inherit the caller's resource-controlled
service and run sequentially.

The caller must supply a frozen attempt key, complete commands, exact environments
and working directory. Its input verifier must check the committed registration,
sources, runtime, data and settings. Its result validator must verify the complete
artifact identity, inventory and bytes; a success string is insufficient. This
executor does not register an experiment, choose a retry, revise diagnostics or
authorize a ranking. An unsuccessful diagnostic report is still a completed
diagnostic operation; the scientific controller must preserve its failed decision.

An existing request cannot change. A recorded process failure is terminal and is
never automatically rerun. After an operating-system interruption, recovery first
checks the saved process identities using Linux boot identity, process identifier
and start ticks. An unreadable process state cannot prove death. A process lock
also prevents concurrent owners. The child inherits that lock through execution;
the parent records its identity before releasing a start barrier. Thus losing the
controller between launch and bookkeeping cannot start an unrecorded fitting
worker. The replacement controller must also account for its original systemd
service before invoking recovery.

A complete sealed worker result can be reconciled after an interruption only
through the original artifact validator. A partial sampler worker resumes its
same-target checkpoints, retaining the original adaptation state, random keys,
settings and draw inventory. Worker-specific checkpoint/archive recovery still
belongs to that worker, not to this generic executor. Completed stages are
revalidated without rerunning them. Invocation logs, previous process identities
and the predecessor receipt hash remain in the execution history. Reconciliation
does not invent a successful exit status for the interrupted invocation: its
unobserved exit remains unobserved. The recovered artifact must independently
establish the declared stage's completion.

This is unselected development infrastructure. Verification is pending. Full-model
worker integration, fixed-retry integration and a registered complete runtime
benchmark remain required; this definition supplies no overnight capability.
