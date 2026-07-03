Place the MediaPipe Pose Landmarker task model used by your working Raspberry Pi Python 3.11 environment here as:

```text
pose_landmarker_lite.task
```


This snapshot now includes `pose_classifier.joblib` trained on the synthetic bootstrap `pose_dataset.csv`. Treat it as a starter model for local testing, not as clinical validation. Retrain it with real camera data from your Raspberry Pi sessions before relying on the model output.

After collecting and training project-specific pose data, `pose_training.py` writes:

```text
pose_classifier.joblib
pose_classifier.report.json
```

The task model was not present in the uploaded source archive and is therefore not bundled here.

An existing model from the original project can remain named `pose_model.pkl` in the
`rpi_agent/` directory. It is loaded through `LEGACY_POSE_CLASSIFIER_MODEL` and does not
need to be renamed to `pose_classifier.joblib`.
