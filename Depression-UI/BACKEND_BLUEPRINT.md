# MindScope Backend Blueprint (Reverse-Engineered from Frontend)

Date: 2026-04-13
Scope: Derived strictly from frontend code in this repository.

## Validation Basis (What Was Observed)

Observed from code:
- Routing and guards in src/App.jsx
- UI flows in src/pages/*
- Local mock data layer in src/services/api.js
- Assessment scoring and severity logic in src/pages/Assessment.jsx and src/data/questionsData.js
- Results/trend rendering and clinician/admin dashboards in src/pages/Results.jsx, src/pages/DoctorDashboard.jsx, src/pages/AdminDashboard.jsx
- Audio recording behavior in src/components/VoiceRecorder.jsx

Not observed:
- No real HTTP network calls (no fetch/axios usage)
- No real backend auth/session protocol
- No file upload to server
- No persistent storage outside browser localStorage/sessionStorage

---

## 1) Feature Extraction (Critical)

## Module A: Public Experience and Navigation

### Feature A1: Landing page and onboarding
Functionality:
- Marketing/intro page with FAQ and CTA.
- CTA routes by auth state and role.

Backend support required:
- Optional CMS/config endpoint for homepage copy/FAQ (if content should be dynamic).
- No hard backend dependency for current static version.

### Feature A2: Contact action
Functionality:
- Contact action is currently mailto.

Backend support required (if upgraded):
- Contact form endpoint with anti-spam protections.

## Module B: Identity and Access

### Feature B1: Role-based signup (patient or doctor)
Functionality:
- Step-based signup.
- Role-specific fields and validations.

Backend support required:
- Registration endpoint with role-specific validation.
- Email uniqueness enforcement.
- Password hashing and policy enforcement.

### Feature B2: User login
Functionality:
- Email/password login.
- Redirect to patient or doctor flow based on role.

Backend support required:
- Login endpoint issuing signed access token (and refresh token ideally).
- Session management and token invalidation.

### Feature B3: Admin login
Functionality:
- Separate admin login flow with admin dashboard.

Backend support required:
- Admin auth endpoint.
- Strong RBAC/ABAC enforcement for admin-only resources.

### Feature B4: Sign out
Functionality:
- Clears current session in frontend.

Backend support required:
- Logout endpoint (token revocation/blacklist for stateful revocation strategy).

## Module C: PHQ-8 Assessment Flow

### Feature C1: Guided 8-question assessment
Functionality:
- Single-question stepper, progress, previous/next navigation.
- Requires recording per question to proceed.

Backend support required:
- Assessment session creation.
- Question set retrieval (versioned).
- Answer submission and finalization endpoint.

### Feature C2: Voice capture and metadata
Functionality:
- Browser microphone capture using MediaRecorder.
- Local preview/playback and duration measurement.
- Frontend maps duration to score (0..3) currently.

Backend support required:
- Secure media upload API (pre-signed URL or direct multipart endpoint).
- Metadata persistence per answer: question_id, duration_sec, codec, size, checksum.
- Optional async transcription/inference pipeline.

### Feature C3: Scoring + severity classification
Functionality:
- Current score derived from duration thresholds and summed to 0..24.
- Severity bucket: Minimal/Mild/Moderate/Moderately Severe/Severe.

Backend support required:
- Authoritative scoring service (server-side).
- Severity logic versioning and audit trail.
- Optional model-based inference support.

## Module D: Processing and Results

### Feature D1: Processing screen
Functionality:
- Simulated processing steps then redirect.

Backend support required:
- Job status endpoint for real async processing.

### Feature D2: Results report
Functionality:
- Displays latest score, severity, interpretation, suggestions.
- Renders trend chart from historical assessments.

Backend support required:
- Assessment result retrieval endpoint.
- Patient history endpoint with pagination and filters.
- Recommendation content endpoint (static or rules-based).

## Module E: Patient History

### Feature E1: Assessment history list
Functionality:
- Patient sees their previous sessions with score/severity/date.

Backend support required:
- Patient-scoped assessments query endpoint.
- Sorting and pagination.

## Module F: Doctor Dashboard

### Feature F1: Risk alerts and monitoring
Functionality:
- Shows severe/moderately severe recent submissions.
- Trend per patient and severity distribution.

Backend support required:
- Doctor-scoped dashboard aggregate endpoint.
- Efficient server-side aggregations by severity, date, patient.
- Access control so doctors only access permitted patient population.

## Module G: Admin Dashboard

### Feature G1: System-wide user and assessment monitoring
Functionality:
- Totals by role and assessment volume.
- User roster and recent assessment activity.

Backend support required:
- Admin metrics endpoint.
- Admin user list endpoint.
- Admin assessment activity endpoint.

---

## 2) API Contract Design

Note: The frontend currently has no real HTTP API calls. The following endpoints are designed from observed behaviors and data requirements.

## 2.1 Authentication and Session

Endpoint: /api/v1/auth/register
Method: POST
Request JSON:
~~~json
{
  "role": "patient",
  "name": "Jane Doe",
  "email": "jane@example.com",
  "password": "StrongPass123!",
  "age": 28,
  "basicInfo": "Short background note"
}
~~~
Response JSON:
~~~json
{
  "user": {
    "id": "usr_123",
    "role": "patient",
    "name": "Jane Doe",
    "email": "jane@example.com",
    "age": 28,
    "basicInfo": "Short background note",
    "createdAt": "2026-04-13T10:00:00Z"
  }
}
~~~
Error cases:
- 400 validation_error (missing/invalid fields)
- 409 email_already_exists
- 422 role_field_mismatch

Endpoint: /api/v1/auth/register
Method: POST
Request JSON (doctor):
~~~json
{
  "role": "doctor",
  "name": "Dr Alex",
  "email": "alex@clinic.com",
  "password": "StrongPass123!",
  "specialization": "Psychiatry",
  "licenseNumber": "MED-548721",
  "clinicName": "City Wellness Hospital",
  "yearsExperience": 8
}
~~~
Response JSON: same shape with doctor fields
Error cases:
- 400 validation_error
- 409 email_already_exists
- 422 license_invalid

Endpoint: /api/v1/auth/login
Method: POST
Request JSON:
~~~json
{
  "email": "jane@example.com",
  "password": "StrongPass123!"
}
~~~
Response JSON:
~~~json
{
  "accessToken": "jwt_access",
  "refreshToken": "jwt_refresh",
  "expiresIn": 3600,
  "user": {
    "id": "usr_123",
    "role": "patient",
    "name": "Jane Doe",
    "email": "jane@example.com"
  }
}
~~~
Error cases:
- 401 invalid_credentials
- 423 account_locked

Endpoint: /api/v1/auth/admin/login
Method: POST
Request JSON:
~~~json
{
  "adminId": "admin@mindscope.ai",
  "password": "Admin@2026!"
}
~~~
Response JSON:
~~~json
{
  "accessToken": "jwt_admin_access",
  "refreshToken": "jwt_admin_refresh",
  "expiresIn": 3600,
  "admin": {
    "id": "adm_1",
    "adminId": "admin@mindscope.ai"
  }
}
~~~
Error cases:
- 401 invalid_admin_credentials
- 403 admin_disabled

Endpoint: /api/v1/auth/logout
Method: POST
Request JSON:
~~~json
{
  "refreshToken": "jwt_refresh"
}
~~~
Response JSON:
~~~json
{
  "success": true
}
~~~
Error cases:
- 401 invalid_token

Endpoint: /api/v1/auth/me
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "user": {
    "id": "usr_123",
    "role": "patient",
    "name": "Jane Doe",
    "email": "jane@example.com"
  }
}
~~~
Error cases:
- 401 unauthorized

## 2.2 Assessment Lifecycle

Endpoint: /api/v1/phq8/questions
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "version": "phq8_v1",
  "questions": [
    { "id": 1, "text": "...", "instruction": "..." }
  ],
  "options": [
    { "label": "Not at all", "value": 0 },
    { "label": "Several days", "value": 1 },
    { "label": "More than half the days", "value": 2 },
    { "label": "Nearly every day", "value": 3 }
  ]
}
~~~
Error cases:
- 500 config_unavailable

Endpoint: /api/v1/assessments
Method: POST
Request JSON:
~~~json
{
  "questionSetVersion": "phq8_v1",
  "answers": [
    {
      "questionId": 1,
      "score": 2,
      "durationSec": 31.2,
      "audioFileId": "file_001"
    }
  ],
  "recordingCount": 8
}
~~~
Response JSON:
~~~json
{
  "assessment": {
    "id": "asmt_001",
    "userId": "usr_123",
    "score": 12,
    "severity": "Moderate",
    "createdAt": "2026-04-13T10:30:00Z"
  }
}
~~~
Error cases:
- 400 insufficient_answers
- 400 invalid_question_id
- 422 invalid_score_range
- 401 unauthorized

Endpoint: /api/v1/assessments/latest
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "assessment": {
    "id": "asmt_001",
    "score": 12,
    "severity": "Moderate",
    "answers": {
      "1": 2,
      "2": 1
    },
    "recordingCount": 8,
    "createdAt": "2026-04-13T10:30:00Z"
  }
}
~~~
Error cases:
- 404 no_assessment
- 401 unauthorized

Endpoint: /api/v1/assessments
Method: GET
Request JSON (query params): page, pageSize, sort
Response JSON:
~~~json
{
  "items": [
    {
      "id": "asmt_001",
      "score": 12,
      "severity": "Moderate",
      "recordingCount": 8,
      "createdAt": "2026-04-13T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "pageSize": 20,
    "total": 56
  }
}
~~~
Error cases:
- 401 unauthorized

## 2.3 Audio Upload and Processing

Endpoint: /api/v1/files/audio/upload-url
Method: POST
Request JSON:
~~~json
{
  "fileName": "q1.webm",
  "contentType": "audio/webm",
  "contentLength": 81234
}
~~~
Response JSON:
~~~json
{
  "fileId": "file_001",
  "uploadUrl": "https://storage/...",
  "expiresIn": 900
}
~~~
Error cases:
- 400 invalid_file_type
- 413 file_too_large
- 401 unauthorized

Endpoint: /api/v1/files/audio/complete
Method: POST
Request JSON:
~~~json
{
  "fileId": "file_001",
  "checksumSha256": "...",
  "durationSec": 31.2
}
~~~
Response JSON:
~~~json
{
  "success": true,
  "file": {
    "id": "file_001",
    "status": "available"
  }
}
~~~
Error cases:
- 400 checksum_mismatch
- 404 file_not_found

Endpoint: /api/v1/assessments/{assessmentId}/processing-status
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "status": "processing",
  "progress": 65,
  "stage": "Preparing Report"
}
~~~
Error cases:
- 404 assessment_not_found
- 401 unauthorized

## 2.4 Doctor Dashboard APIs

Endpoint: /api/v1/doctor/dashboard/summary
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "totals": {
    "patients": 120,
    "assessments": 980,
    "highRiskCases": 75,
    "lowRiskCases": 420
  },
  "severityBreakdown": [
    { "severity": "Severe", "count": 20 },
    { "severity": "Moderately Severe", "count": 55 }
  ]
}
~~~
Error cases:
- 401 unauthorized
- 403 forbidden_non_doctor

Endpoint: /api/v1/doctor/dashboard/alerts
Method: GET
Request JSON (query params): severity[]=Severe&severity[]=Moderately%20Severe&limit=10
Response JSON:
~~~json
{
  "items": [
    {
      "assessmentId": "asmt_001",
      "patient": { "id": "usr_123", "name": "Jane Doe", "email": "jane@example.com" },
      "severity": "Severe",
      "score": 21,
      "createdAt": "2026-04-13T10:30:00Z"
    }
  ]
}
~~~
Error cases:
- 401 unauthorized
- 403 forbidden_non_doctor

Endpoint: /api/v1/doctor/dashboard/patient-trends
Method: GET
Request JSON (query params): patientId optional, range optional
Response JSON:
~~~json
{
  "patients": [
    {
      "patient": { "id": "usr_123", "name": "Jane Doe" },
      "points": [
        { "session": "S1", "score": 12, "severity": "Moderate", "createdAt": "2026-04-01T00:00:00Z" }
      ]
    }
  ]
}
~~~
Error cases:
- 401 unauthorized
- 403 forbidden_non_doctor

## 2.5 Admin Dashboard APIs

Endpoint: /api/v1/admin/dashboard/snapshot
Method: GET
Request JSON: none
Response JSON:
~~~json
{
  "totals": {
    "users": 220,
    "doctors": 35,
    "patients": 185,
    "assessments": 2100
  },
  "users": [
    {
      "id": "usr_123",
      "name": "Jane Doe",
      "email": "jane@example.com",
      "role": "patient",
      "age": 28,
      "basicInfo": "...",
      "createdAt": "2026-04-13T10:00:00Z"
    }
  ],
  "assessments": [
    {
      "id": "asmt_001",
      "userId": "usr_123",
      "userName": "Jane Doe",
      "email": "jane@example.com",
      "score": 12,
      "severity": "Moderate",
      "recordingCount": 8,
      "createdAt": "2026-04-13T10:30:00Z"
    }
  ]
}
~~~
Error cases:
- 401 unauthorized
- 403 forbidden_non_admin

---

## 3) Data Model and Database Design

Recommended DB: PostgreSQL (transactional, relational, robust indexing) + object storage for audio files.

## 3.1 Core Tables

### users
Fields:
- id (uuid, pk)
- role (enum: patient, doctor, admin)
- name (varchar 120, not null)
- email (citext unique, not null)
- password_hash (text, not null)
- age (smallint, nullable; patient)
- basic_info (text, nullable; patient)
- specialization (varchar 120, nullable; doctor)
- license_number (varchar 80, nullable; doctor)
- clinic_name (varchar 160, nullable; doctor)
- years_experience (smallint, nullable; doctor)
- status (enum: active, disabled)
- created_at (timestamptz)
- updated_at (timestamptz)

Indexes:
- unique(email)
- index(role)
- index(created_at desc)
- partial index on (license_number) where role='doctor'

### auth_sessions
Fields:
- id (uuid, pk)
- user_id (uuid, fk users.id)
- refresh_token_hash (text)
- user_agent (text)
- ip_address (inet)
- expires_at (timestamptz)
- revoked_at (timestamptz nullable)
- created_at (timestamptz)

Indexes:
- index(user_id)
- index(expires_at)
- unique(refresh_token_hash)

### question_sets
Fields:
- id (uuid, pk)
- key (varchar 64, unique)  example phq8_v1
- title (varchar 120)
- is_active (boolean)
- created_at (timestamptz)

### questions
Fields:
- id (uuid, pk)
- question_set_id (uuid, fk question_sets.id)
- ordinal (smallint)
- prompt_text (text)
- instruction (text)
- min_score (smallint default 0)
- max_score (smallint default 3)
- created_at (timestamptz)

Indexes:
- unique(question_set_id, ordinal)

### assessments
Fields:
- id (uuid, pk)
- user_id (uuid, fk users.id)
- question_set_id (uuid, fk question_sets.id)
- score_total (smallint)
- severity (enum: Minimal, Mild, Moderate, Moderately Severe, Severe)
- recording_count (smallint)
- scoring_strategy (varchar 64) example duration_threshold_v1
- status (enum: completed, processing, failed)
- created_at (timestamptz)

Indexes:
- index(user_id, created_at desc)
- index(severity, created_at desc)
- index(created_at desc)

### assessment_answers
Fields:
- id (uuid, pk)
- assessment_id (uuid, fk assessments.id)
- question_id (uuid, fk questions.id)
- score (smallint)
- duration_sec (numeric(6,2), nullable)
- audio_file_id (uuid, fk media_files.id nullable)
- created_at (timestamptz)

Indexes:
- unique(assessment_id, question_id)
- index(question_id)

### media_files
Fields:
- id (uuid, pk)
- owner_user_id (uuid, fk users.id)
- storage_provider (varchar 32)
- storage_key (text)
- mime_type (varchar 80)
- bytes_size (bigint)
- duration_sec (numeric(6,2))
- checksum_sha256 (char(64))
- status (enum: pending_upload, available, quarantined, deleted)
- created_at (timestamptz)

Indexes:
- index(owner_user_id, created_at desc)
- index(status)
- unique(checksum_sha256)

### processing_jobs
Fields:
- id (uuid, pk)
- assessment_id (uuid, fk assessments.id)
- job_type (enum: scoring, transcription, inference, report)
- status (enum: queued, running, succeeded, failed)
- progress_pct (smallint)
- attempts (smallint)
- error_code (varchar 64 nullable)
- error_message (text nullable)
- started_at (timestamptz)
- finished_at (timestamptz)
- created_at (timestamptz)

Indexes:
- index(assessment_id)
- index(status, created_at)

### audit_logs
Fields:
- id (bigserial, pk)
- actor_user_id (uuid nullable)
- actor_role (varchar 32)
- action (varchar 80)
- entity_type (varchar 80)
- entity_id (varchar 80)
- metadata (jsonb)
- created_at (timestamptz)

Indexes:
- index(actor_user_id, created_at desc)
- index(entity_type, entity_id)
- brin(created_at) for high-volume retention

## 3.2 Relations

- users 1..N auth_sessions
- users 1..N assessments
- question_sets 1..N questions
- assessments 1..N assessment_answers
- assessment_answers N..1 media_files (optional)
- assessments 1..N processing_jobs

## 3.3 Scalability Considerations

- Use object storage (S3/GCS/Azure Blob) for audio, not DB blobs.
- Use async job queue for heavy inference/transcription.
- Add read replicas for dashboard-heavy aggregate queries.
- Partition assessments by created_at (monthly) when volume grows.
- Precompute dashboard aggregates into materialized views or OLAP sink.

---

## 4) PRD (Product Requirements Document)

## 4.1 Product Overview
MindScope is a depression screening workflow product based on PHQ-8 with role-based access (patient, doctor, admin), voice-assisted response capture, computed severity outcomes, and monitoring dashboards.

## 4.2 Target Users
- Patient: completes assessments and reviews historical trends.
- Doctor: monitors patients, trends, and high-risk alerts.
- Admin: monitors system-wide users and assessment activity.

## 4.3 Key Features
- Role-based registration and login.
- PHQ-8 guided assessment.
- Audio response capture and metadata storage.
- Assessment scoring and severity classification.
- Results report with trend chart.
- Patient history timeline.
- Doctor dashboard with risk alerts and distributions.
- Admin dashboard with global metrics.

## 4.4 User Flows

Patient flow:
1) Register/login as patient
2) Start assessment
3) Record response per question
4) Submit and process
5) View results and trend
6) Review historical sessions

Doctor flow:
1) Login as doctor
2) Open dashboard
3) Review alerts and trends
4) Drill into patient trajectory

Admin flow:
1) Admin login
2) View totals, users, and activity stream

## 4.5 Functional Requirements
- FR-1: System must enforce role-based route access.
- FR-2: Registration must validate role-specific fields.
- FR-3: Assessment must require all 8 responses before completion.
- FR-4: Score and severity must be computed server-side and stored immutably.
- FR-5: Patient can only read own assessments.
- FR-6: Doctor can only access assigned/authorized patient data.
- FR-7: Admin can access system-wide operational views.
- FR-8: All auth-protected APIs require valid bearer token.

## 4.6 Non-Functional Requirements
- Latency:
  - p95 read APIs under 250 ms
  - p95 write APIs under 400 ms (excluding long async jobs)
- Scalability:
  - 10k DAU baseline, horizontal API scale
  - async workers autoscale on queue depth
- Reliability:
  - API uptime >= 99.9%
  - idempotent writes for submit endpoints
- Privacy:
  - encryption at rest and in transit
  - auditable access to health-related records

---

## 5) System Design Document

## 5.1 Textual Architecture Diagram

~~~text
[Web Frontend]
    |
    | HTTPS (JWT)
    v
[API Gateway / BFF]
    |
    +--> [Auth Service] ---------> [PostgreSQL: users, sessions]
    |
    +--> [Assessment Service] ----> [PostgreSQL: assessments, answers]
    |            |
    |            +--> [Object Storage: audio files]
    |            |
    |            +--> [Queue] -> [Processing Worker / ML Service]
    |
    +--> [Dashboard Service] -----> [Read Models / Aggregates]
    |
    +--> [Admin Service] ---------> [Audit Log Store]
~~~

## 5.2 Services
- API server/BFF: routing, validation, auth middleware, response shaping.
- Auth service: registration/login/token refresh/logout.
- Assessment service: create/finalize assessments, answer persistence.
- Media service: upload URL generation and integrity checks.
- Processing/ML service: async scoring/transcription/inference.
- Dashboard service: doctor/admin aggregates and trends.
- Storage layers: PostgreSQL + object storage + optional Redis cache.

## 5.3 Request Flow (Frontend -> Backend -> ML -> Response)
1) Frontend requests upload URL for each audio recording.
2) Frontend uploads audio to object storage.
3) Frontend submits assessment payload with answer metadata and file IDs.
4) Backend persists assessment as processing and enqueues jobs.
5) Worker processes scoring/inference and updates status/result.
6) Frontend polls status endpoint (or subscribes via SSE/WebSocket).
7) Frontend fetches finalized result and renders report.

---

## 6) Edge Cases and Failure Handling

## 6.1 Input/Validation
- Invalid email/password format
- Role mismatch fields (doctor fields sent for patient and vice versa)
- Age under minimum threshold
- Invalid score range outside 0..3
- Missing required answers (<8)

Handling:
- Return structured 400/422 with field-level error details.

## 6.2 Audio/File Failures
- Unsupported codec/mime type
- Empty file or oversized file
- Corrupt upload or checksum mismatch
- Upload URL expired

Handling:
- 400 invalid_file_type
- 413 payload_too_large
- 409 checksum_mismatch
- Retry with fresh signed URL

## 6.3 Processing and ML Failures
- Worker timeout
- Inference service unavailable
- Partial processing failure

Handling:
- Mark job failed with error code
- Exponential backoff retry (max attempts)
- Fallback deterministic scoring path
- Return meaningful status to client

## 6.4 Availability/Timeout Strategy
- API timeout budget example 3 seconds for sync endpoints.
- Long tasks must be async with status endpoint.
- Circuit breaker around ML dependencies.
- Dead-letter queue for repeatedly failing jobs.

---

## 7) Security Design

## 7.1 Authentication
- Access JWT short TTL (for example 15 minutes).
- Refresh token rotation and revocation list.
- Password hashing with Argon2id or bcrypt cost >= 12.

## 7.2 Authorization
- Enforce role checks server-side on every endpoint.
- Patient data scope by user_id.
- Doctor scope by assignment/tenant rules.
- Admin endpoints isolated with strict policies.

## 7.3 File Upload Security
- Only allow trusted MIME and extension combinations.
- Validate max size and duration constraints.
- Malware scanning and quarantine pipeline.
- Signed URL expiry and one-time usage constraints.

## 7.4 API Protection
- Rate limiting by IP + user + route.
- Brute-force protection for login endpoints.
- CSRF protection if cookie auth is used.
- Security headers and strict CORS policy.
- Comprehensive audit logs for PHI/PII access.

## 7.5 Privacy and Compliance Considerations
- Encrypt PII and health-related data at rest.
- TLS 1.2+ enforced in transit.
- Data retention and deletion policies.
- Consent and disclosure flows for voice data usage.

---

## 8) Final Output Summary

## 8.1 Feature List
- Public onboarding and role-aware CTAs
- Patient/doctor signup and login
- Admin login and admin dashboard
- PHQ-8 guided assessment with recording per question
- Processing stage and final report
- Patient assessment history
- Doctor monitoring dashboard with risk insights

## 8.2 API Contract
- Designed full REST contract for auth, assessments, uploads, processing, doctor/admin analytics.
- Clearly separated observed frontend behavior from designed missing APIs.

## 8.3 DB Schema
- Normalized relational schema with users, sessions, assessments, answers, media, jobs, audit logs.
- Includes indexing and scaling strategies.

## 8.4 PRD
- Includes product scope, target users, flows, FR/NFR, performance/reliability goals.

## 8.5 System Design
- Modular service architecture with async processing pipeline for audio/inference workloads.

## 8.6 Risks and Improvements

Primary risks:
- Current frontend scoring is duration-based and not clinically robust.
- No observed backend calls means API payload details are inferred from current local data usage.
- Some route redirection inconsistencies exist in frontend pages (for example doctor/patient guards still using /signin in some pages).

Recommended improvements:
- Introduce strict API schemas (OpenAPI + runtime validation).
- Move scoring to backend and version algorithm.
- Add assignment model for doctor-to-patient authorization.
- Add observability: tracing, metrics, error budgets, SLO dashboard.
- Add automated contract tests from frontend route flows.

---

## Assumptions (Explicit)

1) The product intends production backend replacement for src/services/api.js local mock functions.
2) Audio is intended to be persisted server-side in production.
3) Severity definitions remain PHQ-8 compatible as currently encoded in frontend.
4) Doctor dashboard should operate on authorized patient cohorts, not global unrestricted data.
5) Admin role remains separate from doctor/patient role families.

If any assumption is incorrect, update these first before implementation to avoid contract drift.
