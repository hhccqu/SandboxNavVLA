# Open-loop comparisons: 4,040 identical held-out windows

| Method | N | Path ADE | Path FDE | Time ADE | Time FDE |
|---|---:|---:|---:|---:|---:|
| SandboxNav-VLA (full) | 4040 | 11.09 | 25.45 | 1.40 | 2.92 |
| History disabled at inference | 4040 | 11.50 | 26.17 | 1.46 | 3.12 |
| Navigation removed at inference | 4040 | 14.45 | 34.46 | 1.70 | 3.76 |
| SimLingo (sandbox-adapted, single-frame) | 4040 | 9.63 | 21.98 | 2.44 | 4.33 |

All errors are in cm; lower is better. ADE is weighted by valid points; FDE uses the last valid target per window. Path uses 20 anchors, approximately 5 cm apart (0–95 cm), with the origin excluded from Path ADE. Time uses 10 future waypoints at 0.25 s intervals (0.25–2.50 s).

The two disabled-input rows are inference-time interventions on the same final checkpoint, without retraining. The single-frame reference, when present, is an earlier sandbox-adapted SimLingo checkpoint trained with a different data pool and budget; it is neither unadapted official SimLingo nor a matched-training history ablation. No statistical significance is claimed for these temporally correlated windows.

Maximum same-ID deviation from the archived Full ADE, in cm: {"path_ade_including_origin": 0.0, "time_ade": 0.0}.

## Per-command results (same masks and physical units)

| Method | Command | N | Path ADE | Path FDE | Time ADE | Time FDE |
|---|---|---:|---:|---:|---:|---:|
| SandboxNav-VLA (full) | Continue turning left. | 764 | 8.05 | 17.74 | 1.17 | 2.39 |
| SandboxNav-VLA (full) | Continue turning right. | 502 | 10.21 | 24.20 | 1.36 | 2.63 |
| SandboxNav-VLA (full) | Follow the road. | 1299 | 13.10 | 28.92 | 1.58 | 3.54 |
| SandboxNav-VLA (full) | Go straight through the junction. | 722 | 16.41 | 39.71 | 1.45 | 3.21 |
| SandboxNav-VLA (full) | Turn left at the next junction. | 460 | 5.93 | 12.98 | 1.17 | 2.17 |
| SandboxNav-VLA (full) | Turn right at the next junction. | 293 | 7.16 | 16.78 | 1.43 | 2.57 |
| History disabled at inference | Continue turning left. | 764 | 8.60 | 18.75 | 1.18 | 2.41 |
| History disabled at inference | Continue turning right. | 502 | 10.82 | 25.99 | 1.31 | 2.53 |
| History disabled at inference | Follow the road. | 1299 | 13.89 | 30.38 | 1.71 | 3.89 |
| History disabled at inference | Go straight through the junction. | 722 | 16.46 | 38.77 | 1.60 | 3.65 |
| History disabled at inference | Turn left at the next junction. | 460 | 5.67 | 12.90 | 1.18 | 2.26 |
| History disabled at inference | Turn right at the next junction. | 293 | 7.04 | 17.00 | 1.43 | 2.58 |
| Navigation removed at inference | Continue turning left. | 764 | 13.56 | 32.05 | 1.67 | 3.68 |
| Navigation removed at inference | Continue turning right. | 502 | 18.95 | 42.15 | 2.51 | 5.51 |
| Navigation removed at inference | Follow the road. | 1299 | 12.18 | 28.91 | 1.45 | 3.16 |
| Navigation removed at inference | Go straight through the junction. | 722 | 15.18 | 36.39 | 1.56 | 3.54 |
| Navigation removed at inference | Turn left at the next junction. | 460 | 14.47 | 35.76 | 1.69 | 3.80 |
| Navigation removed at inference | Turn right at the next junction. | 293 | 17.60 | 45.39 | 1.93 | 4.11 |
| SimLingo (sandbox-adapted, single-frame) | Continue turning left. | 764 | 7.55 | 16.90 | 2.43 | 4.20 |
| SimLingo (sandbox-adapted, single-frame) | Continue turning right. | 502 | 10.85 | 25.38 | 1.87 | 3.31 |
| SimLingo (sandbox-adapted, single-frame) | Follow the road. | 1299 | 11.39 | 25.98 | 2.41 | 4.40 |
| SimLingo (sandbox-adapted, single-frame) | Go straight through the junction. | 722 | 9.94 | 23.03 | 2.57 | 4.55 |
| SimLingo (sandbox-adapted, single-frame) | Turn left at the next junction. | 460 | 7.33 | 16.05 | 2.66 | 4.54 |
| SimLingo (sandbox-adapted, single-frame) | Turn right at the next junction. | 293 | 8.29 | 18.34 | 2.89 | 5.14 |
