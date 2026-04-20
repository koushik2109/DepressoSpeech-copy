# MindScope - Depression Screening UI

MindScope is a frontend-first mental health screening application built with React and Vite.
It provides a guided PHQ-8 assessment flow with voice recording, score visualization, and an admin monitoring dashboard.

## Overview

This project currently implements:

- Public landing page and product intro
- Role-based signup (patient/doctor)
- User signin flow
- Admin signin and dashboard
- PHQ-8 assessment with one question per step
- Voice recording per question (browser microphone)
- Processing screen with animated status steps
- Results page with severity meter, chart, and recommendations

Important: the data layer is a mock local API (localStorage/sessionStorage), not a real backend service.

## Tech Stack

- React 19
- Vite 7
- React Router DOM 7
- Tailwind CSS 4 + custom CSS variables/utilities
- Recharts (results chart)
- Framer Motion (installed dependency)
- ESLint 9

Browser APIs used:

- MediaRecorder + getUserMedia (audio capture)
- Canvas API (live waveform)
- localStorage/sessionStorage (persistence)

## Project Structure

Core folders:

- src/pages: route pages (Landing, SignIn, SignUp, Assessment, Processing, Results, AdminLogin, AdminDashboard)
- src/components: shared UI and visualization components
- src/services: local mock API/data service
- src/data: PHQ-8 question and severity helpers
- src/hooks: reusable hooks (audio recorder utility)
- src/layouts: app layout wrappers

## Current App Flow

1. User opens landing page.
2. User signs up or signs in.
3. Signed-in user starts assessment.
4. User records one voice answer for each PHQ-8 question.
5. Assessment is saved locally and redirected to processing screen.
6. User sees final result with score and severity label.
7. Admin can sign in to view totals, user list, and assessment activity.

## Routes

- / - Landing
- /signup - User registration
- /signin - User login
- /assessment - PHQ-8 flow
- /processing - Result preparation transition
- /results - Final report and visualizations
- /admin - Admin login
- /admin/dashboard - Admin monitoring panel

## Mock API Layer

Service file: src/services/api.js

It currently provides frontend-only functions for:

- registerUser
- loginUser
- getCurrentUser
- logoutUser
- saveAssessment
- listAssessments
- getDashboardSnapshot
- loginAdmin
- getAdminSession

Storage keys:

- mindscope-users
- mindscope-assessments
- mindscope-session
- mindscope-admin-session
- latestAssessment (sessionStorage)

Authentication note:

- A mock JWT-like token is generated for UI/session behavior.
- This token is not cryptographically signed and is not production auth.

## Assessment and Scoring

Question source:

- src/data/questionsData.js (PHQ-8 questions, severity helpers)

Current score behavior in UI:

- Each question uses recorded audio duration to map to a value from 0 to 3.
- Total score range: 0 to 24.

Severity mapping:

- 0-4: Minimal
- 5-9: Mild
- 10-14: Moderate
- 15-19: Moderately Severe
- 20-24: Severe

## Development

Install dependencies:

```bash
npm install
```

Run dev server:

```bash
npm run dev
```

Create production build:

```bash
npm run build
```

Preview production build:

```bash
npm run preview
```

Lint:

```bash
npm run lint
```

## UI and Styling Notes

- Global theme variables and custom classes are defined in src/index.css.
- Tailwind extension tokens are configured in tailwind.config.js.
- The visual style uses mint/emerald tones with card-based layouts and animated transitions.

## Known Limits (Current Version)

- No real backend API yet
- No real database
- No secure authentication/authorization
- No server-side validation
- No production-grade medical inference model integration

## Production Upgrade Path

To move this project from prototype to production:

1. Replace local service methods with real HTTP API calls.
2. Add secure auth (signed JWT, refresh tokens, role guards).
3. Move user and assessment storage to a backend database.
4. Add encryption and privacy controls for audio and health data.
5. Integrate real ML/analytics inference pipeline for scoring.
6. Add automated tests (unit, integration, end-to-end).

## Disclaimer

This project is a screening UI prototype.
It does not provide a clinical diagnosis and should not replace professional mental health care.
