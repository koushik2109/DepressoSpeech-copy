const API_BASE = "http://localhost:8000/api/v1";
const SESSION_KEY = "mindscope-session";

// ── HTTP helper ─────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
  const headers = {
    "Content-Type": "application/json",
    "Accept-Encoding": "gzip",
    ...options.headers,
  };
  if (session?.token) {
    headers["Authorization"] = `Bearer ${session.token}`;
  }
  const adminSession = JSON.parse(
    localStorage.getItem("mindscope-admin-session") || "null",
  );
  if (adminSession?.token && !session?.token) {
    headers["Authorization"] = `Bearer ${adminSession.token}`;
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

// ── Auth ────────────────────────────────────────────────

export async function registerUser(userData) {
  const data = await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify(userData),
  });
  return data;
}

export async function loginUser({ email, password }) {
  const data = await apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });

  const session = {
    token: data.accessToken,
    refreshToken: data.refreshToken,
    user: data.user,
  };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

export async function loginAdmin({ adminId, password }) {
  const data = await apiFetch("/auth/admin/login", {
    method: "POST",
    body: JSON.stringify({ adminId, password }),
  });

  const adminSession = {
    token: data.accessToken,
    adminId: data.admin.adminId,
    savedAt: Date.now(),
  };
  localStorage.setItem("mindscope-admin-session", JSON.stringify(adminSession));
  return adminSession;
}

// ── OTP Verification ────────────────────────────────────

export async function verifyOtp({ email, otp }) {
  return apiFetch("/auth/verify-otp", {
    method: "POST",
    body: JSON.stringify({ email, otp }),
  });
}

export async function resendOtp({ email }) {
  return apiFetch("/auth/resend-otp", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

// ── Forgot / Reset Password ────────────────────────────

export async function forgotPassword({ email }) {
  return apiFetch("/auth/forgot-password", {
    method: "POST",
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword({ email, otp, newPassword }) {
  return apiFetch("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ email, otp, newPassword }),
  });
}

// ── Google OAuth ────────────────────────────────────────

export async function googleLogin(credential) {
  const data = await apiFetch("/auth/google", {
    method: "POST",
    body: JSON.stringify({ credential }),
  });

  const session = {
    token: data.accessToken,
    refreshToken: data.refreshToken,
    user: data.user,
  };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  return session;
}

// ── Session helpers ─────────────────────────────────────

// Synchronous – reads from localStorage
export function getCurrentUser() {
  try {
    const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
    return session?.user || null;
  } catch {
    return null;
  }
}

export function updateCurrentUser(updates) {
  try {
    const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
    if (session?.user) {
      session.user = { ...session.user, ...updates };
      localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    }
  } catch {
    /* ignore */
  }
}

export function getAdminSession() {
  try {
    const session = JSON.parse(
      localStorage.getItem("mindscope-admin-session") || "null",
    );
    if (!session) return null;
    // Expire admin sessions after 12 hours to prevent auto-login on next visit
    const SESSION_TTL_MS = 12 * 60 * 60 * 1000;
    if (!session.savedAt || Date.now() - session.savedAt > SESSION_TTL_MS) {
      localStorage.removeItem("mindscope-admin-session");
      return null;
    }
    return session;
  } catch {
    return null;
  }
}

export function logoutUser() {
  localStorage.removeItem(SESSION_KEY);
  localStorage.removeItem("mindscope-admin-session");
}

// ── Assessments ─────────────────────────────────────────

export async function saveAssessment(assessment) {
  const data = await apiFetch("/assessments", {
    method: "POST",
    body: JSON.stringify({
      questionSetVersion: "phq8_v1",
      answers: Object.entries(assessment.answers || {}).map(([qId, score]) => ({
        questionId: Number(qId),
        score: Number(score),
        durationSec: null,
        audioFileId: assessment.audioFileIds?.[qId] || null,
      })),
      recordingCount: assessment.recordingCount || 0,
    }),
  });

  // Return a shape compatible with what the frontend expects
  return {
    id: data.assessment.id,
    userId: data.assessment.userId,
    score: data.assessment.score,
    severity: data.assessment.severity,
    createdAt: data.assessment.createdAt,
    answers: assessment.answers,
    recordingCount: assessment.recordingCount,
    userName: assessment.userName,
    email: assessment.email,
    role: assessment.role,
  };
}

export async function listAssessments() {
  const data = await apiFetch("/assessments?page=1&pageSize=100");
  // Map to the shape the frontend expects
  return (data.items || []).map((a) => ({
    id: a.id,
    score: a.score,
    severity: a.severity,
    recordingCount: a.recordingCount,
    createdAt: a.createdAt,
    userId: getCurrentUser()?.id,
    userName: getCurrentUser()?.name,
    email: getCurrentUser()?.email,
    role: getCurrentUser()?.role,
    mlScore: a.mlScore,
    mlSeverity: a.mlSeverity,
  }));
}

export async function getLatestAssessment() {
  const data = await apiFetch("/assessments/latest");
  return data.assessment;
}

// ── Dashboard ───────────────────────────────────────────

export async function getDashboardSnapshot() {
  const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
  const adminSession = JSON.parse(
    localStorage.getItem("mindscope-admin-session") || "null",
  );

  if (adminSession?.token) {
    try {
      return await apiFetch("/admin/dashboard/snapshot");
    } catch {
      // fallback
    }
  }

  if (session?.user?.role === "doctor") {
    try {
      const summary = await apiFetch("/doctor/dashboard/summary");
      const trends = await apiFetch(
        "/doctor/dashboard/patient-trends?limit=50",
      );
      const alerts = await apiFetch("/doctor/dashboard/alerts?limit=12");

      // Build the snapshot shape the frontend expects
      const assessments = [];
      for (const p of trends.patients || []) {
        for (const pt of p.points || []) {
          assessments.push({
            id: pt.session,
            userId: p.patient.id,
            userName: p.patient.name,
            score: pt.score,
            severity: pt.severity,
            createdAt: pt.createdAt,
          });
        }
      }

      return {
        users: [],
        assessments,
        totals: summary.totals,
        alerts: alerts.items || [],
      };
    } catch {
      // fallback
    }
  }

  return {
    users: [],
    assessments: [],
    totals: { users: 0, doctors: 0, patients: 0, assessments: 0 },
  };
}

// ── PHQ-8 Questions (from backend) ──────────────────────

export async function fetchQuestions() {
  return apiFetch("/phq8/questions");
}

// ── Audio Upload ───────────────────────────────────────

export async function uploadAudio(blob, filename = "recording.webm") {
  const session = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
  const adminSession = JSON.parse(
    localStorage.getItem("mindscope-admin-session") || "null",
  );
  const token = session?.token || adminSession?.token;

  const formData = new FormData();
  formData.append("file", blob, filename);

  const res = await fetch(`${API_BASE}/files/audio/upload`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Upload failed (${res.status})`);
  }
  return res.json();
}

// ── ML Details & Monitoring ────────────────────────────

export async function getProcessingStatus(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/processing-status`);
}

export async function getMLDetails(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/ml-details`);
}

export async function getAdminMetrics() {
  return apiFetch("/admin/dashboard/metrics");
}

export async function getMLHealth() {
  return apiFetch("/admin/dashboard/ml-health");
}
