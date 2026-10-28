# A Survey of Reinforcement Learning for Large Reasoning Models (2025)

Date: 2026-10-28 | Asked by: HanZ

Summary: A large survey (Zhang et al., Tsinghua / Shanghai AI Lab and others, arXiv:2509.08827, v3 dated 9 Oct 2025) of RL applied to LLMs for reasoning, focused on the post-DeepSeek-R1 era. It covers reward design, policy optimization (PPO-style and critic-free methods such as GRPO), sampling strategies, open debates (does RL discover new reasoning or only sharpen existing behavior? does RL generalize where SFT memorizes?), training resources, and applications. The authors argue that scaling RL toward ASI now faces constraints in compute, algorithms, data and infrastructure.

## Main content

- **Framing.** RLHF and DPO aligned models to human preferences. RLVR (RL with verifiable rewards, e.g. answer correctness or unit-test pass rates) is what unlocked long chain-of-thought reasoning in o1 and DeepSeek-R1. The survey calls RL "the foundational methodology" for turning LLMs into large reasoning models (LRMs).
- **Foundational components.** Reward design (verifiable, generative, dense, unsupervised, reward shaping). Policy optimization (policy gradient, critic-based, critic-free such as GRPO, off-policy, regularization). Sampling strategy (dynamic sampling, sampling hyperparameters).
- **Open problems the survey highlights.** RL's role (sharpening vs. discovery); RL vs. SFT (generalize vs. memorize); model prior (how much the base model determines what RL can do); training recipes (tricks vs. traps); process vs. outcome rewards.
- **Resources and applications.** Static corpora, dynamic environments, RL infrastructure (OpenRLHF, veRL, AReaL, slime, TRL). Applications: coding, agentic tasks, multimodal, multi-agent, robotics, medical.
- **Future directions.** Continual RL, memory-based RL, model-based RL, efficient reasoning, latent-space reasoning, RL for pretraining, RL for diffusion LLMs, scientific discovery, architecture-algorithm co-design.

## Speculative (not claimed by the paper)

- The "sharpening vs. discovery" debate is where the most useful follow-up experiments probably live. Measuring pass@k at large k for base vs. RL-tuned models is one cheap way to probe it.

## Sources

- Zhang, K., Zuo, Y., He, B., et al. (2025). *A Survey of Reinforcement Learning for Large Reasoning Models.* arXiv:2509.08827. https://arxiv.org/abs/2509.08827
- Awesome list accompanying the survey: https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs
