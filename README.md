# Entropy collapse in RLVR

A small, self-contained notebook that reproduces **entropy collapse** in reinforcement
learning with verifiable rewards (RLVR) — and the covariance mechanism behind it — in a
setting tiny enough to check by hand.

Instead of a full language model, the policy here is a softmax over `K` possible answers
trained with a faithful miniature of **GRPO** (Group Relative Policy Optimization). This
keeps every quantity inspectable while still showing the same qualitative behaviour
reported for real RLVR runs.

## What's inside

The notebook (`RLVR_Entropy_experiment.ipynb`) walks through four experiments:

1. **Watch it collapse.** Policy entropy falls smoothly and monotonically from `log K`
   toward zero while performance rises — nothing in the objective asks for this; it is a
   side effect of the updates.
2. **The mechanism is a covariance.** The entropy change per step equals a covariance
   between log-probability and advantage, and that covariance stays positive, so entropy
   can only fall. The measured and predicted entropy changes match almost exactly.
3. **The law.** Performance and entropy track `R = b - a·exp(H)` across seeds
   (`R² ≈ 0.98`), which implies a hard performance ceiling as entropy runs out.
4. **When collapse hurts, and what fixes it.** On a task with a tempting mediocre trap,
   plain GRPO collapses early and never finds the optimum. A naive entropy bonus does not
   help (matching the literature), while the targeted **Clip-Cov** method does — it removes
   the high-covariance, collapse-driving updates without blocking ordinary exploitation.

## Running it

```bash
pip install numpy matplotlib
jupyter notebook RLVR_Entropy_experiment.ipynb
```

## Paper

This notebook reproduces and explores the central result of:

> Cui et al., *The Entropy Mechanism of Reinforcement Learning for Reasoning Language
> Models*, arXiv:2505.22617 (2025) — https://arxiv.org/abs/2505.22617
> ([reference code](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL))

The "clip-higher" idea referenced in Experiment 4 comes from Yu et al., *DAPO*.
