# Combining neurons makes Qwen 'become' Cookie Monster

Asked by HanZ on 2026-10-28.

A poster on X (treetown) reports finding individual neurons in Qwen with small effects on their own, but larger and sometimes surprising effects when activated together. Their headline example: combining a "Cookie" neuron with a "Cartoon mascot" neuron makes Qwen mention Cookie Monster, while either neuron alone never did. Why it's cool: it hints at compositional, non-additive structure in neuron-level features, which is a question interp people care about. It is also a single-person, anecdotal report, so treat it as a lead, not a result.

## Details

- The thread is labelled (1/3) and the replies we could read continue it. Adding Genki and Monster neurons reportedly makes the model describe itself as Cookie Monster.
- Activation method, model size, and how neurons were selected are not in the visible text.
- Views and replies are recorded on the post, but no independent reproduction was found.

## Why it matters (speculative)

- If combinations of neurons behave non-additively, single-neuron labels could miss most of what a model does. That fits the superposition picture, but this post alone does not show it.
- Good next step: check the claim with activation steering on a fixed set of prompts, measure the rate of "Cookie Monster" mentions with and without each neuron, and test on a second model.

## Sources

- treetown (@treetowntree), X post, 2026-10-26: https://x.com/treetowntree/status/2114573600976183723
