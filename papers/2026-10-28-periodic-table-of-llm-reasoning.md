# The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes (2026)

Date: 2026-10-28 | Asked by: HanZ

Summary: A systematic-review-style survey (arXiv:2606.11470v1) of more than 300 papers on reasoning in LLMs. It organizes the literature into nine reasoning paradigms (chain-of-thought, multi-hop, mathematical, commonsense, visual/temporal, code/algorithmic, retrieval-augmented, tool-augmented/agentic, and RL-based), compares methods and evaluation benchmarks across them, and catalogues recurring failure modes such as hallucinated reasoning traces, brittle multi-step inference and poor cross-domain generalization. It proposes meta-reasoning, self-evolving reasoning, multimodal reasoning and socially grounded reasoning as open directions.

## Main content

- **Method.** Keyword-driven search across several databases, then inclusion/exclusion filters and eight quality-assessment criteria (rated poor to excellent). Preprints were included only if widely cited or needed for very recent developments.
- **Main thesis.** Reasoning abilities in LLMs are real but fragile. They depend heavily on prompting, task structure and scale. The authors argue that final-answer accuracy is the dominant metric, which hides whether a model reasoned or guessed.
- **Recommendations.** Step-level (process) supervision, reward models that score reasoning quality, verification-oriented training, hybrid neuro-symbolic and tool-augmented architectures, and unified evaluation of depth, coherence and generalization.

## Notes and caveats

- It is a survey: no new experiments of its own.
- The abstract lists Semantic Scholar, Google Scholar, Papers with Code and the ACL Anthology, while the methodology section lists a somewhat different set of databases. Worth keeping in mind when judging coverage.
- Speculative: the "periodic table" framing is a taxonomy metaphor. It organizes the field, but it does not predict which combinations of paradigms will work.

## Sources

- The Periodic Table of LLM Reasoning: A Structured Survey of Reasoning Paradigms, Methods, and Failure Modes (2026). arXiv:2606.11470v1. https://arxiv.org/html/2606.11470v1
