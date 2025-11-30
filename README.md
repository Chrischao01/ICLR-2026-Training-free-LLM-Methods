
# ICLR-2026-Training-free-LLM-Methods

This repo tracks my personal project to **reproduce and analyze all training-free (or training-light) LLM methods submitted to ICLR 2026**.

The goal is not just to get numbers, but to understand *what these methods are really doing* and whether we can extract some common principles for steering and analyzing LLMs/VLMs.

---

## Why I’m Doing This

My current belief is:

> Many “training-free” or “steering” methods are actually exposing behaviors that are usually buried inside parameter updates.

So I’m asking myself:

* Can we **abstract and organize** the existing methods into a small set of reusable analysis patterns for LLMs/VLMs?
* Can we understand:

  * their **entry points** (where they intervene: logits, attention, residual, memory, etc.),
  * their **motivation** (what failure modes they target),
  * their **implementation details** (what they actually modify),
  * and their **application scenarios** (alignment, reasoning, routing, safety, etc.)?

This repo is my attempt to build a **systematic map** of current training-free methods:

* reproduce results,
* read and annotate code,
* and summarize what seems to consistently work (and what doesn’t).

Comments/issues/PRs are very welcome if you’re working on similar things.

---

## Current Paper List (ICLR 2026 Submissions)

I’m gradually adding reproduction notes, code pointers, and short takeaways for each method.

| Paper                                                                                                             | Short Description                                                                 | Review Info*                                     |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------ |
| **COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics**                          | In-context steering via one-step dynamics.                                        | Ratings: [4, 6, 6, 8] · Conf: [3, 2, 3, 3]       |
| **In-Context Prompt Optimisation for Knowledge Editing: Enhancing Safety and Coherency in Large Language Models** | Training-free-ish prompt optimization for safer knowledge editing.                | Ratings: [2, 2, 2, 4, 4] · Conf: [3, 4, 3, 3, 3] |
| **Decoupled Alignment for Robust Plug-and-Play Adaptation**                                                       | Plug-and-play alignment via decoupled components.                                 | Ratings: [2, 4, 4, 6] · Conf: [4, 3, 4, 2]       |
| **ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training**             | Initial-token attention steering (head-wise), supervised + unsupervised variants. | Ratings: [4, 4, 6, 6] · Conf: [4, 3, 3, 5]       |
| **Toward Preference-aligned Large Language Models via Residual-based Model Steering**                             | Residual-based steering for preference alignment.                                 | Ratings: [2, 2, 4] · Conf: [4, 4, 4]             |
| **Fast Intent Classification for LLM Routing via Statistical Analysis of Representations**                        | Representation-based routing / intent classification.                             | Ratings: [2, 2, 4, 8] · Conf: [3, 4, 2, 4]       |
| **Command-V: Training-Free Representation Finetuning Transfer**                                                   | Training-free representation transfer for V/LLMs.                                 | Ratings: [6, 6, 6, 10] · Conf: [3, 3, 4, 5]      |
| **Training-Free Group Relative Policy Optimization**                                                              | Group-based test-time policy adjustment without training.                         | Ratings: [2, 2, 4, 6] · Conf: [4, 4, 5, 3]       |
| **SinkTrack: Attention Sink based Context Anchoring for Large Language Models**                                   | Uses attention sinks for context anchoring and robustness.                        | Ratings: [4, 4, 4, 6, 6] · Conf: [3, 2, 4, 3, 3] |
| **Plug-and-Play Global Memory via Test-Time Registers**                                                           | Adds a global memory via test-time registers, no full fine-tuning.                | Ratings: [2, 4, 4, 4] · Conf: [4, 3, 2, 3]       |

*Review info is just for reference and may not reflect final decisions.

---

## Roadmap

Planned steps for this repo:

1. **Reproduce core results** for each paper (or as close as possible given resources).
2. **Unify the interface**: a common runner / config format to compare methods side by side.
3. **Abstract patterns**:

   * where they intervene (attention, logits, residuals, memory, routing),
   * what signal they use (entropy, margins, preferences, sinks, etc.).
4. **Write up notes**: short summaries, failure cases, and cross-method comparisons.

If you’re also exploring training-free steering, feel free to open an issue or PR with your own notes, scripts, or additional papers to include.
