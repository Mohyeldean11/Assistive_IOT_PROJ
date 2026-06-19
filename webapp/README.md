# CareAgent AI Web App

Responsive local Flask dashboard and REST API for one Raspberry Pi device and one monitored person.

See the repository-level `README.md` for complete installation, Raspberry Pi integration, mobile access, and safety notes.

Primary device endpoints:

```text
GET  /health
POST /api/readings
POST /api/readings/batch
```

Device requests require the `X-API-Key` header. The application stores readings in SQLite and appends accepted payloads to `logs/readings.json`.
