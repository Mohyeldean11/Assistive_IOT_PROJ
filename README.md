# CareAgent AI — Enhanced Local Elderly Monitoring Prototype

**Developer & Architect: Mohie Eldean Badawy**

This package combines:

- A responsive Flask web application for a single monitored person.
- A Raspberry Pi 5 agent designed for **Python 3.11**.
- Reliable JSON/JSONL telemetry forwarding over the local network.
- SQLite storage, JSON audit logs, admin/user authentication, email alerts, and a mobile phone-call action.
- A confidence-aware MediaPipe pose pipeline with temporal smoothing and explicit `UNKNOWN` rejection.
- A conservative camera-based stroke-warning and emergency engine.
- A safer pose training workflow with session-aware evaluation.

Cloud integration is intentionally disabled and isolated in `future_cloud/`.

## Project layout

```text
careagent_ai_enhanced/
├── webapp/                  # Local-PC Flask dashboard and API
├── rpi_agent/               # Python 3.11 Raspberry Pi monitoring agent
│   ├── camera_pipeline.py   # Persistent C930/C930e + MediaPipe VIDEO mode
│   ├── pose_classifier.py   # Rules/model ensemble + temporal hysteresis
│   ├── stroke_detector.py   # Persistent visual warning signals
│   ├── emergency_engine.py  # Emergency decision logic
│   ├── telemetry_client.py  # Reliable uploader with retry spool
│   ├── log_forwarder.py     # Sends legacy JSON/JSONL logs to the web app
│   └── pose_training.py     # Collection, training, and evaluation
└── future_cloud/            # Commented placeholders only
```

## 1. Run the web app on the local PC

```bash
cd webapp
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

The server listens on all local interfaces at port 5000. On the PC, open:

```text
http://127.0.0.1:5000
```

On a phone connected to the same Wi-Fi network, open:

```text
http://YOUR_PC_LOCAL_IP:5000
```

Allow inbound TCP port 5000 in the PC firewall. Do not expose this development server directly to the public internet.

Default accounts are configured in `.env.example`. Change the passwords, `SECRET_KEY`, and `DEVICE_API_KEY` before use.

## 2. Configure the Raspberry Pi 5 agent

Use the existing Python 3.11 environment in which your MediaPipe installation already works. The official MediaPipe PyPI package does not consistently provide Linux ARM64 wheels, so this project deliberately does not replace or pin your working Pi installation.

```bash
cd rpi_agent
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-py311.txt
cp .env.example .env
```

Edit `.env`:

```text
CAREAGENT_URL=http://YOUR_PC_LOCAL_IP:5000/api/readings
DEVICE_API_KEY=the-same-key-used-by-webapp
SIMULATE_SENSORS=true
```

Place your existing MediaPipe task model here:

```text
rpi_agent/models/pose_landmarker_lite.task
```

Run the agent:

```bash
python main.py
```

The camera is analyzed continuously at the configured analysis interval. Telemetry is uploaded every **120 seconds**, while confirmed emergency patterns can trigger an earlier upload. Failed uploads are saved to `rpi_agent/spool/` and retried later.

## 3. Forward existing JSON logs

The forwarder accepts:

- A JSON array of payloads.
- One JSON payload object.
- A `.jsonl` file containing one payload per line.

```bash
python log_forwarder.py /path/to/old_logs.json
```

It creates deterministic event IDs for legacy records, remembers the last forwarded index, and relies on the web app’s duplicate protection.

## 4. Mobile emergency actions

The responsive dashboard provides two separate actions:

- **Call contact:** opens the phone dialer through a `tel:` link when the dashboard is viewed on a mobile device.
- **SOS email alert:** sends the configured emergency email and records the alert attempt in SQLite.

A web browser cannot silently place a phone call. The user must confirm the call in the mobile dialer.

## 5. Why the pose pipeline is less likely to hallucinate

The original code opened and closed the camera for every frame and used independent image inference. The enhanced pipeline:

- Keeps the Logitech camera open, avoiding repeated autofocus/auto-exposure resets.
- Uses MediaPipe `VIDEO` mode, which enables frame-to-frame tracking.
- Rejects dark, overexposed, or excessively blurred frames.
- Rejects poses with insufficient landmark visibility.
- Normalizes measurements by body scale instead of relying only on raw image coordinates.
- Combines rule and trained-model predictions.
- Requires minimum model probability.
- Uses multi-frame voting and hysteresis.
- Returns `UNKNOWN` instead of guessing when evidence is weak.

`LYING` is used instead of claiming `LAYING ON BED` or `LAYING ON FLOOR`, because body landmarks alone cannot reliably identify the supporting surface. A calibrated room/bed region or a separate scene model would be needed for that distinction.

## 6. Train the pose classifier

This package includes a deterministic synthetic bootstrap dataset at `rpi_agent/pose_dataset.csv` and a matching starter model at `rpi_agent/models/pose_classifier.joblib`. These samples are for pipeline testing only; replace or augment them with real Raspberry Pi captures before presenting the model as reliable.

Regenerate the bootstrap dataset and starter model:

```bash
cd rpi_agent
python pose_training.py --generate-synthetic --synthetic-per-label 160 --synthetic-sessions 4 --train
```

Collect real data in several independent sessions rather than collecting one long sequence and randomly splitting adjacent frames.

```bash
python pose_training.py --collect STANDING --samples 200
python pose_training.py --collect SITTING --samples 200
python pose_training.py --collect LYING --samples 200
python pose_training.py --train
```

The trained bundle and report are written to:

```text
models/pose_classifier.joblib
models/pose_classifier.report.json
```

Collection guidance:

- Record multiple healthy volunteers when possible.
- Vary clothing, lighting, background, camera distance, and body orientation.
- Keep the whole body visible and avoid placing the person at the extreme edges of the 90-degree frame.
- Collect separate sessions on different days so the holdout set is more realistic.
- Never ask an elderly or at-risk person to perform a fall. Stage dangerous classes only with healthy adults, supervision, and safe padding, or use ethically obtained annotated footage.
- Inspect the confusion matrix and low-confidence fraction before accepting the model.

## 7. Important algorithm changes

- DHT22 temperature is treated as **room temperature**, not body temperature.
- Abnormal room temperature/humidity creates an environment warning, not a medical emergency by itself.
- PIR “no motion” alone does not trigger an emergency because a person may be sleeping or outside the PIR cone.
- Facial and arm asymmetry require visible landmarks and persistence across frames.
- Collapse plus subsequent immobility carries the highest emergency weight.
- The optional Ollama module only summarizes structured evidence. It cannot alter pose, risk, or emergency decisions.

## 8. Tests

Web app:

```bash
cd webapp
pytest -q
```

Raspberry Pi logic tests:

```bash
cd rpi_agent
pytest -q
```

Camera and MediaPipe hardware behavior must still be validated on the actual Raspberry Pi 5 and Logitech C930/C930e.

## 9. Security note

The uploaded legacy Azure file contained a live-looking storage connection string. It has been removed from this project. Rotate/revoke that Azure storage key before any future cloud trial. Never commit connection strings, API keys, SMTP passwords, or device keys.

## Medical limitation

This is an experimental monitoring aid, not a certified medical device and not a stroke diagnosis system. Camera pose landmarks do not fully assess speech, vision, headache, sensation, clinical history, or all balance symptoms. Suspected stroke or immediate danger requires direct assessment and local emergency services without waiting for this software.

## Reference material

- Google MediaPipe Pose Landmarker Python documentation: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- CDC stroke signs and B.E. F.A.S.T.: https://www.cdc.gov/stroke/signs-symptoms/index.html
- Logitech C930e product information: https://www.logitech.com/en-us/products/webcams/c930e-business-webcam.html


## Camera Snapshot Upload

The Raspberry Pi agent now attaches one compressed JPEG snapshot to each periodic or emergency telemetry upload. The Flask web app saves the image under `webapp/static/snapshots/`, replaces the raw base64 data with an image URL, and refreshes the dashboard image automatically.

RPi `.env` options:

```env
ENABLE_SNAPSHOT_UPLOAD=true
SNAPSHOT_JPEG_QUALITY=70
SNAPSHOT_MAX_WIDTH=960
```

Keep `SNAPSHOT_MAX_WIDTH` around 640-960 for the Raspberry Pi 5 demo. Larger images increase latency and request size without improving stroke-risk decisions. The image is for visual confirmation only; emergency logic still comes from sensor and pose-analysis data.

## Pose regression fix (snapshot release)

The snapshot feature does not create a second camera stream. It reuses the latest frame
from the persistent camera pipeline. This release fixes a compatibility issue where an
unset/zero MediaPipe landmark `presence` field or a heuristic image-quality warning could
cause all valid landmarks to be discarded and the uploaded pose to remain `UNKNOWN`/`None`.

Run this on the Raspberry Pi before starting the full agent:

```bash
cd rpi_agent
source .venv/bin/activate
python pose_diagnostic.py --frames 30
```

A healthy output should normally show `landmarks=33`, `features=yes`, and a non-UNKNOWN
`raw`/`stable` pose after the person is fully visible. The first accepted pose is now
reported immediately, while temporal voting still smooths later changes.

The original `pose_model.pkl` is also supported. Place it at `rpi_agent/pose_model.pkl`
or set `LEGACY_POSE_CLASSIFIER_MODEL` to its actual path. The deterministic classifier
continues to work when no compatible trained model is present.
