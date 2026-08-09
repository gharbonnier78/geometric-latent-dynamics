# Session handoff — Geometric Latent Dynamics

**Origin:** ChatGPT / MachineLearning project, 2026-08-09  
**Stable session title:** **LLPNN — Geometric Latent Dynamics, Deep Reading & Pedagogical Harness**  
**Repository:** `gharbonnier78/geometric-latent-dynamics`

## Source paper

Vakhtang Putkaradze, **“Latent Lie-Poisson Neural Networks (LLPNNs): Discovering the motion of Lie-Poisson systems through observable data and latent dynamics”**, arXiv:2607.28939v1, 31 Jul 2026.

Expected source PDF:

- path: `papers/2607.28939v1.pdf`
- uploaded source SHA-256: `e0a8335592744be93eb7e3d9bc88840ce53fd73736b1388f7b82698d6a1b1051`

The workflow is designed to commit the arXiv v1 PDF only if its SHA-256 matches the originally uploaded source exactly.

## Why this repository exists

The repository is deliberately broader than a reproduction of LLPNN.

Its target is a reproducible path from conventional ML / basic linear algebra to advanced geometric latent dynamics, with two simultaneous assurance tracks:

- **Research assurance:** claims, hypotheses, evidence, experiments, reproducibility, gates.
- **Pedagogical assurance:** prerequisites, intuition, mathematical descent, plain-language interpretation, executable examples, misconception checks, understanding gates.

The source paper is the **scientific target, not the table of contents**.

## Central conceptual result to teach first

The key insight extracted from the paper is stronger than “measurements are noisy”:

> **Observable variables may fail to form a dynamical state at all.**

In the degenerate `SE(2)` example:

\[
h(\mu)=\mu_1+\frac12\mu_3^2,
\]

\[
\xi=\nabla h(\mu)=
\begin{pmatrix}
1\\0\\\mu_3
\end{pmatrix},
\]

while

\[
\dot\mu_3=\mu_2.
\]

Therefore \(\mu_2\) is hidden from the current observable \(\xi\), yet it controls the immediate future of \(\mu_3\), hence of the future observable.

Suggested Study 0 pair:

\[
\mu^{(A)}=(1,+1,0.5)^T,
\qquad
\mu^{(B)}=(1,-1,0.5)^T.
\]

Both yield

\[
\xi^{(A)}=\xi^{(B)}=(1,0,0.5)^T,
\]

but

\[
\dot\mu_3^{(A)}=+1,
\qquad
\dot\mu_3^{(B)}=-1.
\]

So:

\[
\boxed{\text{same observation} \not\Rightarrow \text{same latent state} \not\Rightarrow \text{same future}.}
\]

## Pedagogical contract

Every difficult concept should pass through five stages:

1. **En français dans le texte** — intuitive meaning before jargon.
2. **Concrete example before abstraction.**
3. **Mathematical descent** — derive the expression line by line instead of dropping a formula.
4. **Immediate plain-language interpretation** after the equations.
5. **Executable experiment** that shows what works and what breaks when the structure is ignored.

For every new symbol, answer:

1. What is it?
2. Why do we need it?
3. Where does the expression come from mathematically?
4. What happens if we do not respect it?

## Planned mathematical progression

Do **not** start with Lie-Poisson.

0. Why Euclidean updates are insufficient for constrained geometric states.
1. Vectors, matrices and transformations — targeted refresh only.
2. Manifolds and tangent spaces, starting with the circle \(S^1\).
3. Groups and Lie groups: \(SO(2)\to SO(3)\to SE(2)\to SE(3)\).
4. Lie algebra and exponential map.
5. Why the dual \(\mathfrak g^*\) appears.
6. Hamiltonian mechanics, Poisson and Lie-Poisson.
7. Casimirs, then Noether as separate pedagogical steps.
8. Study 0: same observation, different latent state, different future on \(SE(2)\).
9. Only then implement the neural network / LLPNN.

## LLPNN reading conclusions already established

### What the paper does

- assumes the symmetry group / Lie-Poisson structure is known;
- observes configuration and reduced velocity variables;
- treats momentum variables as latent;
- learns either a Hamiltonian decoder or a pseudo-Lagrangian encoder on active variables;
- uses a universal Noether invariant to reconstruct latent trajectories;
- advances dynamics using Lie-Poisson flows and Magnus-based Lie-group updates;
- preserves Casimirs by construction / coadjoint-orbit evolution.

### Most important mathematical difficulty

For degenerate Hamiltonians, the Legendre transform can be non-invertible. The observable velocity can therefore lose latent coordinates that still affect future dynamics. A naïve autonomous model \(\dot\xi=f(\xi)\) need not exist.

### What is structurally guaranteed vs not established generally

Structurally supported in the paper:

- Casimir preservation under the geometric update;
- latent reconstruction through the Noether relation under the paper’s symmetry assumptions;
- applicability of the Hamiltonian formulation to degenerate Hamiltonians.

Not generally established:

- unique physical identification of the latent momentum (gauge ambiguity remains);
- general observability of trajectory-specific \(p_0\);
- global NN optimization convergence;
- arbitrary observation operators;
- arbitrary Poisson manifolds;
- automatic discovery of the symmetry group.

## Relation to the wider research axis

Keep these layers distinct:

\[
\text{geometry}
\neq
\text{state inference}
\neq
\text{dynamics learning}
\neq
\text{decision}.
\]

Possible longer-term stack:

\[
\text{observations}
\rightarrow
p(x_t\mid y_{1:t})
\rightarrow
\text{structured latent dynamics}
\rightarrow
\text{action/value}.
\]

LLPNN is **not** itself a Bayesian uncertainty model and does not replace EKF / particle filtering / probabilistic inference.

A useful epistemic warning for ESC/BOS work is:

> runtime evidence can be correct and observable without constituting a sufficient dynamical state.

That is an extrapolation from the paper, not a claim made by the LLPNN paper itself.

## Immediate next work item

Create `experiments/study_0_same_observation_different_future/` before any neural-network implementation.

Study 0 should include:

- a plain-language statement of the phenomenon;
- derivation of the `SE(2)` equations used;
- hand calculation for two latent states;
- a minimal executable simulation;
- plot showing identical initial observable and divergent future observables;
- research-assurance claim/hypothesis/gate;
- pedagogical-assurance prerequisite/intuition/derivation/misconception/understanding gates;
- explicit link back to the relevant equations / lemma in the source paper.

## Resume prompt

A future chat can restart with:

> Open `gharbonnier78/geometric-latent-dynamics`, read `handoff/SESSION_HANDOFF.md` and `handoff/CHAT_TRANSCRIPT_2026-08-09.md`, verify `papers/2607.28939v1.pdf`, then continue with Study 0. Preserve the dual research + pedagogical assurance harness and introduce every mathematical notion through derivation plus plain-language interpretation.
