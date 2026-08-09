# Geometric Latent Dynamics

> **Same observation does not necessarily mean same state — or same future.**

A pedagogical and reproducible lab for hidden-state dynamics on manifolds and Lie groups, from first principles to structure-preserving neural networks.

## Purpose

This repository starts from the paper **Vakhtang Putkaradze, “Latent Lie-Poisson Neural Networks (LLPNNs): Discovering the motion of Lie-Poisson systems through observable data and latent dynamics”, arXiv:2607.28939v1 (31 Jul 2026)**.

The goal is not merely to reproduce LLPNN results. The repository will rebuild the required mathematics progressively and experimentally, assuming only conventional ML foundations and an early exposure to geometry/manifolds.

Every difficult notion should be introduced through the same pedagogical descent:

1. plain-language intuition;
2. concrete example;
3. mathematical derivation, step by step;
4. immediate “en français dans le texte” interpretation of the equations;
5. executable experiment;
6. misconception / understanding gate.

The research harness is therefore also a **pedagogical assurance harness**.

## Initial research question

The first experiment precedes any neural network:

> Can two systems have the same current observable state but different hidden states, and therefore different futures?

The paper’s degenerate `SE(2)` example is the initial target:

\[
h(\mu)=\mu_1+\frac12\mu_3^2,
\qquad
\xi=\nabla h(\mu)=
\begin{pmatrix}1\\0\\\mu_3\end{pmatrix},
\qquad
\dot\mu_3=\mu_2.
\]

Two states with the same \(\mu_3\) but different \(\mu_2\) therefore produce the same current observable \(\xi\) and different immediate futures.

## Repository bootstrap

- [`papers/2607.28939v1.pdf`](papers/2607.28939v1.pdf) — exact source paper studied in the originating ChatGPT session; populated by a checksum-gated GitHub Action.
- [`handoff/CHAT_TRANSCRIPT_2026-08-09.md`](handoff/CHAT_TRANSCRIPT_2026-08-09.md) — full visible transcript of the originating discussion.
- [`handoff/SESSION_HANDOFF.md`](handoff/SESSION_HANDOFF.md) — stable re-entry point: source, decisions, pedagogical contract and next experiment.
- [`handoff/CHAT_TITLE.md`](handoff/CHAT_TITLE.md) — stable chat label plus note about the unavailable exact UI title.

## Planned structure

```text
geometric-latent-dynamics/
├── foundations/
├── pedagogy/
├── experiments/
│   ├── study_0_same_observation_different_future/
│   ├── study_1_so2/
│   ├── study_2_se2/
│   └── ...
├── llpnn/
├── research_assurance/
├── pedagogical_assurance/
├── papers/
└── handoff/
```

## Assurance model

```text
Research Assurance
├── Claim
├── Hypothesis
├── Evidence
├── Experiment
├── Reproducibility
└── Gate

Pedagogical Assurance
├── Prerequisite
├── Intuition
├── Mathematical derivation
├── Plain-language explanation
├── Executable example
├── Misconception check
└── Understanding gate
```

The source paper is the scientific target, not the table of contents. The repository should make the path to that target auditable, executable and teachable.