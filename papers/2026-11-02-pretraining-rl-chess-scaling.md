# Understanding Reasoning from Pretraining to Post-Training

2026-11-02 · asked by arush

A controlled study that pretrains Qwen3-architecture models from 5M to 1B parameters on human chess games, then runs SFT on synthetic reasoning traces and GRPO on chess puzzles with binary rewards. Post-RL pass@1 is well predicted by pretraining loss, and the local RL slope grows roughly linearly with log pretraining tokens. RL mostly amplifies moves the SFT policy already preferred on easy puzzles, but can surface near-absent correct moves on hard ones. A 1B OLMo-2 math check shows the same direction.

## Setup
- Pretraining: 54B tokens of Lichess Blitz/Rapid games, 10 model sizes (5M to 1B), 11 IsoFLOP budgets (6.5e16 to 6.5e19 FLOPs).
- SFT: synthetic reasoning traces built as serialized game trees sampled from the pretrained model. SFT with traces improved pass@8 and pass@16. SFT without traces only improved pass@1.
- RL: GRPO, binary reward (every move in the solution line must match). Runs of 1000 to 5000 steps from SFT checkpoints at 20M, 50M, 200M and 680M.
- Math check: 1B OLMo-2, pretrained to several token budgets (10B to 200B tokens), SFT on NuminaMath-CoT, then GRPO on GSM8K + MATH + DeepScaler.

## Findings
- Compute frontier: for a fixed model, the best split starts at high pretraining fraction (RL is initialization-limited). The RL share rises with total compute; for the 20M model it goes from about 5% to 32%.
- pass@16 is mixed: the 20M model's pass@16 jumps early in RL, but larger models' pass@16 stays flat or slightly degrades.
- Scaling fit: post-RL pass@1 tracks pretraining validation loss (Spearman |rho| 0.93 to 0.99 as reference compute rises). The slope B(N,T) correlates with log10 T (Pearson r = +0.84). Joint fit weights tokens more than parameters.
- Mechanism: a power-sharpening fit gives alpha around 1.0 to 1.35 (global) during RL, but explains only moderate variance. RL reshapes the distribution in state-dependent ways.
- Extrapolated frontier (speculative, from their fitted law): compute-optimal RL share rises with total compute while pretraining tokens stay close to Chinchilla.

## What it does not settle
- It does not report a minimum model size where RL stops working. The smallest sizes are 5M to 20M, and I did not verify the per-size RL outcomes for the 5M and 10M runs.
- Chess is not natural language or arithmetic, and the math check is qualitative.
- The paper is about pass@1 and the pretraining-to-RL interface. Whether RL induces new patterns or only sharpens them remains contested (Yue et al. 2025 vs. Yuan et al. 2025, both cited by the paper).

## Speculative next step
Run the same sweep on a natural-language or arithmetic task (for example multi-step Countdown-style equations), with difficulty bins, and extend below 20M to find where pass@1 under RL leaves the zero-reward floor.

## Sources
- Understanding Reasoning from Pretraining to Post-Training, arXiv:2607.16097 (2026). https://arxiv.org/abs/2607.16097
- Code: https://github.com/pavelslab-nyu/pre2post-chess
- Models and datasets: https://huggingface.co/collections/pavelslab-nyu/pre2post-chess
- Scaling Behaviors of LLM RL Post-Training, arXiv:2509.25300 (2025). https://arxiv.org/abs/2509.25300
- Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (Yue et al., NeurIPS 2025). https://neurips.cc/virtual/2025/poster/119944
