# pregrasp_eval

This directory contains evaluation entry points for pregrasp-initialized
DemoGrasp policies.

The old DGA/generative-pregrasp bridge has been removed from the release tree.
The maintained path is the `PregraspPrior` field/yaw/scorer workflow:

```text
PregraspPrior/data/rollouts/   raw rollout labels
PregraspPrior/data/fields/     aggregated pregrasp fields
PregraspPrior/data/models/     learned scorer checkpoints
```

## Current Entrypoints

`eval_pregrasp_field_SE_residual.py`

Evaluate SE residual PPO with object-level M/T/P fields from
`PregraspPrior/data/fields`.

`eval_yaw_field_SE_residual.py`

Evaluate yaw-expanded field initialization.

`eval_yaw_scorer_SE_residual.py`

Evaluate a learned yaw pregrasp scorer checkpoint.

## Legacy Note

The removed DGA bridge used settled scene export, external generative pregrasp
outputs, and ShadowHand visualization helpers. It is no longer part of the
main release workflow.
