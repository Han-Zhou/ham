# Smallest reasoning-capable model under RL: experiment protocol (draft v3)

2026-11-02 · asked by arush

A draft protocol for finding N*, the smallest model size at which RL on a verifiable arithmetic task gives a reliable pass@1 gain, at Chinchilla-style token budgets. It is a living document: the setup and parameters will change, so each change should be recorded here with its date. Items marked [done] are settled, [open] still need a decision, and [speculative] marks my own estimates.

## 1. Question
At Chinchilla-optimal pretraining, what is the smallest model size N* at which RL on a verifiable task gives a reliable gain in pass@1? Below N*, RL should leave the policy at the zero-reward floor, or give gains within noise.

## 2. Hypothesis (falsifiable)
The verdict is monotone in size: once RL works at N, it works at every larger N in the grid. If the verdict flips back and forth, N* is undefined and we report the flip pattern.

## 3. Data (pretraining corpus) [partly done]
- [done] Corpus: one team-gathered corpus, fixed for all runs. Check licensing and remove personal data before any release. The corpus itself is not published in this repo.
- [done] Split first: carve out the held-out test set and eval probes before anything else.
- [done] Tokenizer: trained on the train split only. Vocabulary fixed across all sizes. [open] Size: small (about 8k to 16k) for natural text, to keep the embedding cost low.
- [done] Dedupe: document-level, including near-duplicates, before sharding.
- [done] Shuffle and interleave: global shuffle with a logged seed. With multiple sources, interleave at fixed weights so every prefix has the same mix.
- [done] Nested shards: one tokenized stream in shuffled order. Model i reads the first T_i tokens. Shards are flat uint16 binaries, with an end-of-document token between documents.
- [open] Coverage: the corpus must hold at least T_max = 20 * N_max tokens. For a grid up to N = 1B, that is 20B tokens, about 40 GB of uint16 storage. If the corpus is smaller, cut the grid or gather more, because repeated data breaks the assumptions here.
- [open] Per-prefix checks: token counts per source, length histograms, and loss of a fixed probe model on each prefix.

## 4. Model grid [partly done]
- [done] Architecture: dense decoder, pre-norm RMSNorm, RoPE, SwiGLU, grouped-query attention, tied embeddings. Fixed across sizes.
- [done] Size definition: N = non-embedding parameters. Embedding size is fixed across the grid, so it cannot drive the threshold.
- [done] Token budget: T_i = 20 * N_i. Shard i = first T_i tokens of the stream.
- [open] Grid: 8 log-spaced sizes, e.g. 5M, 10M, 20M, 50M, 100M, 200M, 500M, 1B. Vary width at roughly fixed depth-to-width ratio (d/L about 64 to 128), one knob at a time. [speculative] The ratio is a heuristic, not a law.

## 5. Task and pipeline [open items marked]
- Task: Countdown-style arithmetic, binary reward. SFT and RL data are generated separately from the pretraining corpus, so the shard scheme does not apply to them.
- SFT: synthetic reasoning traces from a fixed proposal policy, the same for all sizes.
- RL: GRPO, fixed step budget and learning-rate schedule, same prompt set for all sizes. [open] Step count.
- Eval: pass@1 and pass@8 on at least 1,500 held-out puzzles, disjoint from SFT and RL data. Bins B1 to B4 for eval, B5 held out for analysis.

## 6. Verdict criterion (pre-register before any run) [open]
Reasoning is present at N if the RL-minus-SFT pass@1 gain is at least delta, with a 95% bootstrap CI excluding 0. Starting guess for delta: 2 points. The RL checkpoint's pass@1 must also be above the zero-reward floor on at least B2.

N* = the smallest N in the grid where the verdict is "present" and every larger N is also "present".

## 7. Search
- Stage A: all sizes, 1 seed. Mark the first flip.
- Stage B: 3 to 5 seeds at the bracketing sizes, narrowing until the bracket is adjacent grid points.
- Stage C: fit gain vs. log N over all runs; report N* with fit uncertainty.

If Stage B and Stage C disagree, report both.

## 8. Controls
- Second shuffle seed at 2 to 3 sizes near the threshold, to check that a lucky prefix is not setting the result.
- Second token budget (5 * N) to see how N* moves with data.
- Second test distribution (B4 only, or multiplication) to check that N* is not specific to Countdown.

## 9. Compute [speculative estimate]
- Pretraining is cheap for small N. For N = 1B and T = 20B, 6ND gives about 1.2e20 FLOPs.
- RL dominates. The chess paper (arXiv:2607.16097) reports about 160 H200 GPU-hours for 2,000 RL steps on a 50M model. Larger models cost more per step.
- First pass: 8 sizes at 1 seed, then 3 to 5 seeds at 2 to 3 bracketing sizes. That is roughly 20 to 30 RL runs. Run a pilot to check real cost before committing.
- [proposed, see section 14] Pretrain once per size and branch the RL seeds from that checkpoint. This would cut pretraining to one run per size.

## 10. Division of work [open]
- Data pipeline: corpus, tokenizer, dedupe, shards, per-prefix checks.
- Model and training: grid configs, pretraining runs.
- SFT, RL and reward code.
- Eval harness, verdict criterion, bootstrap and curve fitting, write-up.

Everyone reviews sections 6 and 7 before any run.

## 11. Pilot checklist
- Data: shard i equals the first T_i tokens of the stream (checksum). Per-prefix mix checks pass. Test set is disjoint from all training data.
- One size, one seed, end to end. The RL reward is not all zeros. If it is, the task is too hard at that size and the curriculum needs to be easier.
- Eval noise: re-run one checkpoint with a different eval seed.

## 12. What would change our minds
- The verdict flips more than once across the grid: monotonicity fails, so report the flip pattern.
- Gains are present at every size, including the smallest: the floor is below the grid, so extend it downward.
- Gains are absent at every size: the task is too hard, not the model too small.

## 13. Limitations
Countdown is not natural language, so a small-model result does not transfer automatically to math or general reasoning. The grid and seed count are small, so the threshold is coarse.

## 14. Engineering settings [open, proposed]
These are implementation choices. Settings marked "exact" should not change results, up to floating-point noise. Settings marked "model change" change the model, so they are ablations, not defaults. Every setting that changes numerics must be the same across the whole grid.

Exact (proposed as defaults):
- Mixed precision bf16, with fp32 for loss and optimizer states. Same for all sizes.
- FlashAttention for attention.
- Sequence packing with proper attention masking.
- Gradient accumulation to hit the target batch size.
- Gradient checkpointing only if memory requires it.
- Memory-mapped shards with prefetch and pinned memory.
- Fused optimizer and norm kernels, checked once against the reference on a few batches.
- Vocabulary size and hidden dimensions as multiples of 64, chosen at config time.

Numerics-changing (choose once, apply to whole grid, record in changelog):
- torch.compile or CUDA graphs. Small models are often launch-bound, so this can help a lot.
- Deterministic kernels, if affordable. Full determinism can cost speed.

Experiment-structure savings (proposed):
- Pretrain once per size. Branch SFT and RL seeds from that checkpoint, so only RL seeds vary. Check this does not hide variance we need for the verdict.
- Cache SFT traces and held-out eval outputs.
- Use vLLM or a similar engine for RL rollouts.
- Checkpoint often, so preempted runs resume.
- Stop dead RL runs early only under a pre-registered rule, for example zero reward for N consecutive steps.

Ablation arms (model changes, not defaults):
- Activation shift from Peng, Zheng and Liu (arXiv:2610.14500, 2026). The shift subtracts a constant b inside the MLP. Because the down-projection W2 trains, this is not a fixed bias that can be folded away in advance, so it must be implemented directly. How b is set for SwiGLU is undefined in the paper as I read it, so this arm is blocked until that is decided.

## Changelog
- v3 (2026-11-02): added section 14 (engineering settings), and the checkpoint-sharing proposal in section 9.
- v2 (2026-11-02): added data-side decisions (split, tokenizer, dedupe, nested shards, coverage), non-embedding N, and fixed token budget T = 20N.

## Sources
- Understanding Reasoning from Pretraining to Post-Training, arXiv:2607.16097 (2026). https://arxiv.org/abs/2607.16097
- Rethinking Activation Functions: Reducing Bias Can Improve Learning, Peng, Zheng and Liu, arXiv:2610.14500 (2026). https://arxiv.org/abs/2610.14500
- Scaling Behaviors of LLM RL Post-Training, arXiv:2509.25300 (2025). https://arxiv.org/abs/2509.25300
- A Technical Study into Small Reasoning Language Models, arXiv:2506.13404 (2025). https://arxiv.org/abs/2506.13404
- Small Models Struggle to Learn from Strong Reasoners, arXiv:2502.12143 (2025). https://arxiv.org/abs/2502.12143
- TinyZero, Jiayi Pan (2025). https://github.com/Jiayi-Pan/TinyZero
- Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yue et al., NeurIPS 2025). https://neurips.cc/virtual/2025/poster/119944
- Training Compute-Optimal Large Language Models (Hoffmann et al., NeurIPS 2022). https://proceedings.neurips.cc/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf
