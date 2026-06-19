# Pose regression fix

This version keeps snapshot upload enabled and fixes the `pose: None` / permanently
`UNKNOWN` behavior introduced by the stricter pose pipeline.

## Corrected behavior

- MediaPipe inference runs even when a frame quality heuristic reports blur/darkness.
- Landmark `presence` is treated as optional; a valid `visibility` value is enough when
  `presence` is unset or zero.
- The first valid deterministic pose is reported immediately instead of waiting for the
  stabilizer vote window before the first telemetry upload.
- Pose values are normalized so Python `None`, `INIT`, and unsupported labels are never
  transmitted as a JSON null value.
- Original 17-feature `pose_model.pkl` files are supported through
  `LEGACY_POSE_CLASSIFIER_MODEL`.
- `pose_diagnostic.py` prints the complete path from landmarks to final pose.

## Raspberry Pi check

```bash
cd rpi_agent
source .venv/bin/activate
python pose_diagnostic.py --frames 30
```

Expected when the full body is visible:

```text
landmarks=33 frame_usable=True features=yes ... raw=STANDING stable=STANDING
```

A frame may say `frame_usable=False` while still producing a pose. The quality field is
now diagnostic only and no longer erases MediaPipe landmarks.
