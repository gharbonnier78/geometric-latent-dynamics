# Scientific Research Harness reference

Status: **active development reference**

This repository is a consumer of the reusable pedagogical / research-assurance patterns developed in:

- repository: `gharbonnier78/scientific-research-harness`
- pull request: `#1 — Add step-state pedagogy and bootstrap three-view case study`
- branch: `agent/step-state-pedagogy`
- pinned harness commit at this handoff: `a4a3f6a5637baedf7b42ab092b4b096682334105`

## Reused generic contracts

The LLPNN project should reuse, not independently reinvent, at least these upstream rules:

- `pedagogy/STEP_STATE_SPEC.md`
- `pedagogy/PEDAGOGICAL_CONCEPT_CONTRACT.md`
- `design/HARNESS_REUSE_MODEL.md`

## Local LLPNN extensions

The following remain specific to this repository:

- the Putkaradze LLPNN source paper and provenance;
- Lie-group / Lie-algebra / dual-space / Lie-Poisson mathematical derivations;
- the `SE(2)` same-observation/different-future Study 0;
- LLPNN-specific notebooks and experiments;
- concept instances and understanding gates for geometric dynamics;
- research claims and evidence specific to this study.

## Upstream rule

When work in this repository reveals a rule that would benefit unrelated rigorous studies without depending on Lie-Poisson mechanics, propose it upstream to `scientific-research-harness` rather than keeping a silent local fork.

Conversely, LLPNN-specific scientific content must stay here and must not turn the generic harness into a geometric-ML-specific framework.

## Scientific authority boundary

The pedagogical harness explains and structures learning. It does not change scientific claims or evidence.

In particular:

- **scientific gates** decide claim / experiment admissibility;
- **understanding gates** decide only whether the pedagogical progression should advance.

A pedagogical failure cannot turn a scientifically valid result into an invalid one, and pedagogical clarity cannot rescue a scientific gate failure.

## Future stabilization

PR #1 is currently draft. Once the relevant upstream rules are merged and/or tagged, replace this development reference with an immutable merged commit or release tag. Preserve the old ref in history rather than silently pretending the project always used the later harness version.