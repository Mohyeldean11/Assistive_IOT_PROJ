from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import os
import smtplib
import threading
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from functools import wraps
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import LoginManager, UserMixin, current_user, login_required, login_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("careagent.web")

JSON_LOG_LOCK = threading.Lock()
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message_category = "warning"


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Reading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(80), unique=True, nullable=False, index=True)
    device_id = db.Column(db.String(80), nullable=False, index=True)
    captured_at = db.Column(db.String(80), nullable=True)
    received_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    payload_json = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False, default="UNKNOWN")
    emergency = db.Column(db.Boolean, nullable=False, default=False)

    @property
    def payload(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.payload_json)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(50), nullable=True)


class AlertEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    source = db.Column(db.String(120), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    message = db.Column(db.Text, nullable=False)
    email_sent = db.Column(db.Boolean, nullable=False, default=False)


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if current_user.role != "admin":
            abort(403)
        return view(*args, **kwargs)

    return wrapped


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{BASE_DIR / 'instance' / 'monitor.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        # A dashboard snapshot JPEG may be included in the reading JSON.
        MAX_CONTENT_LENGTH=3 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        REMEMBER_COOKIE_HTTPONLY=True,
        REMEMBER_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    (BASE_DIR / "instance").mkdir(exist_ok=True)
    (BASE_DIR / "logs").mkdir(exist_ok=True)
    (BASE_DIR / "static" / "snapshots").mkdir(parents=True, exist_ok=True)
    db.init_app(app)
    login_manager.init_app(app)

    with app.app_context():
        db.create_all()
        ensure_schema_compatibility()
        seed_defaults()

    register_routes(app)
    register_template_helpers(app)
    return app


def ensure_schema_compatibility() -> None:
    """Small SQLite migration path for the original CareAgent prototype database."""
    inspector = inspect(db.engine)
    if "reading" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("reading")}
    with db.engine.begin() as connection:
        if "event_id" not in columns:
            connection.execute(text("ALTER TABLE reading ADD COLUMN event_id VARCHAR(80)"))
            connection.execute(text("UPDATE reading SET event_id = 'legacy-' || id WHERE event_id IS NULL"))
        if "captured_at" not in columns:
            connection.execute(text("ALTER TABLE reading ADD COLUMN captured_at VARCHAR(80)"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_reading_event_id ON reading (event_id)"))


def seed_defaults() -> None:
    admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "Admin@123")
    user_username = os.getenv("DEFAULT_USER_USERNAME", "viewer")
    user_password = os.getenv("DEFAULT_USER_PASSWORD", "User@123")

    if not User.query.filter_by(username=admin_username).first():
        admin = User(username=admin_username, role="admin")
        admin.set_password(admin_password)
        db.session.add(admin)
    if not User.query.filter_by(username=user_username).first():
        viewer = User(username=user_username, role="user")
        viewer.set_password(user_password)
        db.session.add(viewer)
    if not EmergencyContact.query.first():
        db.session.add(
            EmergencyContact(
                name=os.getenv("EMERGENCY_CONTACT_NAME", "Emergency Contact"),
                email=os.getenv("EMERGENCY_CONTACT_EMAIL", "contact@example.com"),
                phone=os.getenv("EMERGENCY_CONTACT_PHONE", ""),
            )
        )
    db.session.commit()


def register_template_helpers(app: Flask) -> None:
    @app.template_filter("clean_status")
    def clean_status(value: Any) -> str:
        return str(value or "Unknown").replace("'", "").replace("_", " ")

    @app.context_processor
    def inject_csrf():
        return {"csrf_token": get_csrf_token}


def get_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = uuid.uuid4().hex
        session["csrf_token"] = token
    return token


def validate_csrf() -> None:
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    if not submitted or not expected or not hmac.compare_digest(submitted, expected):
        abort(400, description="Invalid form token. Refresh the page and try again.")


def _number(value: Any, field: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not low <= number <= high:
        raise ValueError(f"{field} is outside the accepted range")
    return number


def validate_payload(data: dict[str, Any]) -> str | None:
    required = ["deviceId", "dht22", "PIR501", "pose", "health_status", "emergency", "stroke_risk"]
    missing = [key for key in required if key not in data]
    if missing:
        return f"Missing required fields: {', '.join(missing)}"
    if str(data.get("deviceId", "")).strip() != os.getenv("MONITORED_DEVICE_ID", "rpi-01"):
        return "Unexpected deviceId for this single-patient deployment"
    if not isinstance(data.get("dht22"), dict) or not isinstance(data.get("PIR501"), dict):
        return "dht22 and PIR501 must be JSON objects"
    try:
        _number(data["dht22"].get("temperature_celsius"), "temperature_celsius", -40, 80)
        _number(data["dht22"].get("humidity_percent"), "humidity_percent", 0, 100)
    except ValueError as exc:
        return str(exc)
    if data["PIR501"].get("value") not in (0, 1, False, True):
        return "PIR501.value must be 0 or 1"
    risk = data.get("stroke_risk")
    if not isinstance(risk, dict) or "risk_level" not in risk:
        return "stroke_risk must be an object containing risk_level"
    if str(risk.get("risk_level", "")).upper() not in {"LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"}:
        return "Unsupported stroke risk level"
    if not isinstance(data.get("emergency"), bool):
        return "emergency must be true or false"
    snapshot = data.get("snapshot")
    if snapshot is not None:
        if not isinstance(snapshot, dict):
            return "snapshot must be an object"
        if snapshot.get("image_base64") and snapshot.get("mime_type", "image/jpeg") not in {"image/jpeg", "image/jpg", "image/png"}:
            return "snapshot.mime_type must be image/jpeg or image/png"
    return None



def persist_snapshot_if_present(payload: dict[str, Any]) -> dict[str, Any]:
    """Decode an optional base64 JPEG/PNG snapshot and replace it with a static URL.

    The raw image bytes are intentionally not stored in SQLite/readings.json to keep
    the audit log readable and small. The saved file is still linked from the payload.
    """
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("image_base64"):
        return payload
    mime_type = str(snapshot.get("mime_type", "image/jpeg")).lower()
    extension = ".jpg" if mime_type in {"image/jpeg", "image/jpg"} else ".png" if mime_type == "image/png" else None
    if extension is None:
        raise ValueError("snapshot.mime_type must be image/jpeg or image/png")
    try:
        raw = base64.b64decode(str(snapshot.get("image_base64")), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("snapshot.image_base64 is not valid base64") from exc
    if not raw or len(raw) > 1_500_000:
        raise ValueError("snapshot image is empty or too large")
    if extension == ".jpg" and not raw.startswith(b"\xff\xd8"):
        raise ValueError("snapshot does not look like a JPEG image")
    if extension == ".png" and not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("snapshot does not look like a PNG image")

    device_id = secure_filename(str(payload.get("deviceId", "device"))) or "device"
    event_id = secure_filename(str(payload.get("event_id", uuid.uuid4()))) or uuid.uuid4().hex
    snapshots_dir = BASE_DIR / "static" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{device_id}_{event_id}{extension}"
    latest_filename = f"latest_{device_id}{extension}"
    (snapshots_dir / filename).write_bytes(raw)
    (snapshots_dir / latest_filename).write_bytes(raw)

    cleaned = dict(payload)
    cleaned_snapshot = dict(snapshot)
    cleaned_snapshot.pop("image_base64", None)
    cleaned_snapshot.update({
        "filename": filename,
        "latest_filename": latest_filename,
        "image_url": url_for("static", filename=f"snapshots/{filename}"),
        "latest_image_url": url_for("static", filename=f"snapshots/{latest_filename}"),
        "stored_at": datetime.now(timezone.utc).isoformat(),
    })
    cleaned["snapshot"] = cleaned_snapshot
    return cleaned

def append_json_log(payload: dict[str, Any]) -> None:
    log_path = BASE_DIR / "logs" / "readings.json"
    with JSON_LOG_LOCK:
        try:
            existing = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []
        existing.append(payload)
        temp_path = log_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(log_path)


def send_email_alert(subject: str, body: str) -> tuple[bool, str]:
    contact = EmergencyContact.query.first()
    if not contact or not contact.email:
        return False, "Emergency contact email is not configured"
    host = os.getenv("SMTP_HOST", "").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = os.getenv("SMTP_FROM", username).strip()
    if not all([host, username, password, sender]):
        return False, "SMTP settings are incomplete"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = contact.email
    msg.set_content(body)
    try:
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(msg)
        return True, "Email alert sent"
    except Exception as exc:  # SMTP errors differ by provider.
        logger.exception("Email delivery failed")
        return False, f"Email failed: {type(exc).__name__}"


def trigger_alert(payload: dict[str, Any], source: str = "Automatic monitor") -> tuple[bool, str]:
    risk = payload.get("stroke_risk", {}) if isinstance(payload.get("stroke_risk"), dict) else {}
    risk_level = str(risk.get("risk_level", "Unknown")).upper()
    subject = f"URGENT: CareAgent alert for {payload.get('deviceId', 'patient')}"
    signs = [name.replace("_", " ").title() for name, value in risk.items() if isinstance(value, bool) and value]
    body = (
        f"Alert source: {source}\n"
        f"Device: {payload.get('deviceId', 'Unknown')}\n"
        f"Health status: {payload.get('health_status', 'Unknown')}\n"
        f"Risk level: {risk_level}\n"
        f"Detected signs: {', '.join(signs) if signs else 'None listed'}\n"
        f"Captured at: {payload.get('captured_at') or payload.get('dht22', {}).get('timestamp', 'Unknown')}\n"
        f"Received at: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\n\n"
        "This software is a monitoring aid, not a diagnosis. Check the person immediately and call local emergency services for suspected stroke or immediate danger."
    )
    ok, message = send_email_alert(subject, body)
    db.session.add(
        AlertEvent(
            source=source,
            severity=risk_level if risk_level in {"HIGH", "CRITICAL"} else "EMERGENCY",
            message=message,
            email_sent=ok,
        )
    )
    db.session.commit()
    return ok, message


def store_reading(data: dict[str, Any]) -> tuple[Reading, bool, tuple[bool, str] | None]:
    event_id = str(data.get("event_id") or uuid.uuid4())
    data = dict(data)
    data["event_id"] = event_id
    data = persist_snapshot_if_present(data)
    risk_level = str(data["stroke_risk"].get("risk_level", "UNKNOWN")).upper()
    reading = Reading.query.filter_by(event_id=event_id).first()
    if reading:
        return reading, False, None

    reading = Reading(
        event_id=event_id,
        device_id=str(data["deviceId"]),
        captured_at=str(data.get("captured_at") or data.get("dht22", {}).get("timestamp") or ""),
        payload_json=json.dumps(data, ensure_ascii=False),
        risk_level=risk_level,
        emergency=bool(data["emergency"]),
    )
    db.session.add(reading)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        existing = Reading.query.filter_by(event_id=event_id).first()
        if existing:
            return existing, False, None
        raise

    append_json_log(data)
    alert_required = bool(data["emergency"]) or risk_level in {"HIGH", "CRITICAL"}
    email_result = trigger_alert(data) if alert_required else None
    return reading, True, email_result


def check_device_key() -> bool:
    supplied_key = request.headers.get("X-API-Key", "")
    expected_key = os.getenv("DEVICE_API_KEY", "replace-with-a-device-api-key")
    return bool(supplied_key and hmac.compare_digest(supplied_key, expected_key))


def register_routes(app: Flask) -> None:
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "careagent-ai", "time": datetime.now(timezone.utc).isoformat()})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))
        if request.method == "POST":
            validate_csrf()
            user = User.query.filter_by(username=request.form.get("username", "").strip()).first()
            if user and user.check_password(request.form.get("password", "")):
                login_user(user, remember=bool(request.form.get("remember")))
                return redirect(url_for("dashboard"))
            flash("Invalid username or password.", "danger")
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        validate_csrf()
        logout_user()
        return redirect(url_for("login"))

    @app.get("/")
    @login_required
    def dashboard():
        latest = Reading.query.order_by(Reading.received_at.desc()).first()
        recent = Reading.query.order_by(Reading.received_at.desc()).limit(30).all()
        alerts = AlertEvent.query.order_by(AlertEvent.created_at.desc()).limit(6).all()
        contact = EmergencyContact.query.first()
        return render_template("dashboard.html", latest=latest, recent=recent, alerts=alerts, contact=contact)

    @app.post("/api/readings")
    def receive_reading():
        if not check_device_key():
            return jsonify({"error": "Unauthorized device"}), 401
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "JSON object expected"}), 400
        error = validate_payload(data)
        if error:
            return jsonify({"error": error}), 400
        reading, created, email_result = store_reading(data)
        return jsonify({"status": "stored" if created else "duplicate", "id": reading.id, "email_alert": email_result}), 201 if created else 200

    @app.post("/api/readings/batch")
    def receive_batch():
        if not check_device_key():
            return jsonify({"error": "Unauthorized device"}), 401
        batch = request.get_json(silent=True)
        if not isinstance(batch, list) or not 1 <= len(batch) <= 100:
            return jsonify({"error": "Expected a JSON array containing 1 to 100 readings"}), 400
        results = []
        for index, data in enumerate(batch):
            if not isinstance(data, dict):
                results.append({"index": index, "status": "rejected", "error": "Object expected"})
                continue
            error = validate_payload(data)
            if error:
                results.append({"index": index, "status": "rejected", "error": error})
                continue
            reading, created, _ = store_reading(data)
            results.append({"index": index, "status": "stored" if created else "duplicate", "id": reading.id})
        return jsonify({"results": results}), 207

    @app.get("/api/latest")
    @login_required
    def latest_reading():
        reading = Reading.query.order_by(Reading.received_at.desc()).first()
        if not reading:
            return jsonify({"reading": None})
        return jsonify(
            {
                "reading": {
                    "id": reading.id,
                    "received_at": reading.received_at.isoformat(),
                    "payload": reading.payload,
                }
            }
        )

    @app.post("/emergency")
    @login_required
    def emergency():
        validate_csrf()
        latest = Reading.query.order_by(Reading.received_at.desc()).first()
        payload = latest.payload if latest else {
            "deviceId": os.getenv("MONITORED_DEVICE_ID", "rpi-01"),
            "health_status": "Manual emergency button pressed",
            "stroke_risk": {"risk_level": "UNKNOWN"},
        }
        ok, message = trigger_alert(payload, source=f"Manual dashboard SOS by {current_user.username}")
        flash(message, "success" if ok else "warning")
        return redirect(url_for("dashboard"))

    @app.route("/admin/contact", methods=["GET", "POST"])
    @admin_required
    def contact_settings():
        contact = EmergencyContact.query.first()
        if request.method == "POST":
            validate_csrf()
            contact.name = request.form.get("name", "").strip()
            contact.email = request.form.get("email", "").strip()
            contact.phone = request.form.get("phone", "").strip()
            if not contact.name or not contact.email:
                flash("Name and email are required.", "danger")
                return render_template("contact.html", contact=contact), 400
            db.session.commit()
            flash("Emergency contact updated.", "success")
            return redirect(url_for("contact_settings"))
        return render_template("contact.html", contact=contact)

    @app.errorhandler(400)
    def bad_request(error):
        return render_template("error.html", code=400, message=getattr(error, "description", "Bad request.")), 400

    @app.errorhandler(403)
    def forbidden(_):
        return render_template("error.html", code=403, message="Administrator access is required."), 403


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False, threaded=True)
