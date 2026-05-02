const API_BASE = "/api/v1";
const SESSION_KEY = "mindscope-session";
const ADMIN_SESSION_KEY = "mindscope-admin-session";
const SESSION_EVENT = "mindscope-session-updated";

// Auth is intentionally tab-scoped. The previous localStorage approach kept the
// last user's token around and caused shared-browser auto-login.
const sessionStore =
  typeof window !== "undefined" ? window.sessionStorage : null;
const legacyStore = typeof window !== "undefined" ? window.localStorage : null;

if (legacyStore) {
  legacyStore.removeItem(SESSION_KEY);
  legacyStore.removeItem(ADMIN_SESSION_KEY);
}

function readJson(storage, key) {
  if (!storage) return null;
  try {
    return JSON.parse(storage.getItem(key) || "null");
  } catch {
    return null;
  }
}

function writeJson(storage, key, value) {
  if (!storage) return;
  storage.setItem(key, JSON.stringify(value));
}

function notifySessionChange() {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(SESSION_EVENT));
  }
}

// ── Lightweight in-memory response cache ────────────────
// Prevents duplicate network calls when components re-mount rapidly.
const _cache = new Map();
const CACHE_TTL_MS = 15_000; // 15 seconds

function getCached(key) {
  const entry = _cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) {
    _cache.delete(key);
    return null;
  }
  return entry.data;
}

function setCache(key, data) {
  _cache.set(key, { data, ts: Date.now() });
}

/** Invalidate cache entries whose key starts with `prefix`. */
export function invalidateCache(prefix = "") {
  if (!prefix) {
    _cache.clear();
    return;
  }
  for (const k of _cache.keys()) {
    if (k.startsWith(prefix)) _cache.delete(k);
  }
}

// ── In-flight dedup ─────────────────────────────────────
// If the same GET request is already in flight, return the existing promise.
const _inflight = new Map();

// ── HTTP helper ─────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const { skipCache = false, ...fetchOptions } = options;
  const session = readJson(sessionStore, SESSION_KEY);
  const headers = {
    "Content-Type": "application/json",
    ...fetchOptions.headers,
  };
  if (session?.token) {
    headers["Authorization"] = `Bearer ${session.token}`;
  }
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
  if (adminSession?.token && !session?.token) {
    headers["Authorization"] = `Bearer ${adminSession.token}`;
  }

  const method = (fetchOptions.method || "GET").toUpperCase();
  const cacheKey = `${method}:${path}`;
  const canUseCache = method === "GET" && !skipCache;

  // For GET requests: use cache + dedup
  if (canUseCache) {
    const cached = getCached(cacheKey);
    if (cached) return cached;

    if (_inflight.has(cacheKey)) return _inflight.get(cacheKey);
  }

  const fetchPromise = (async () => {
    const res = await fetch(`${API_BASE}${path}`, { ...fetchOptions, headers });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    return res.json();
  })();

  if (canUseCache) {
    _inflight.set(cacheKey, fetchPromise);
    try {
      const data = await fetchPromise;
      setCache(cacheKey, data);
      return data;
    } finally {
      _inflight.delete(cacheKey);
    }
  }

  // Non-GET: invalidate related caches
  const data = await fetchPromise;
  // After mutations, clear all cached GETs so the UI picks up new data
  _cache.clear();
  notifySessionChange();
  return data;
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
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  writeJson(sessionStore, SESSION_KEY, session);
  _cache.clear();
  notifySessionChange();
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
  sessionStore?.removeItem(SESSION_KEY);
  writeJson(sessionStore, ADMIN_SESSION_KEY, adminSession);
  _cache.clear();
  notifySessionChange();
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
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  writeJson(sessionStore, SESSION_KEY, session);
  _cache.clear();
  notifySessionChange();
  return session;
}

// ── Session helpers ─────────────────────────────────────

// Synchronous – reads from tab-scoped sessionStorage
export function getCurrentUser() {
  const session = readJson(sessionStore, SESSION_KEY);
  return session?.user || null;
}

export function updateCurrentUser(updates) {
  const session = readJson(sessionStore, SESSION_KEY);
  if (session?.user) {
    session.user = { ...session.user, ...updates };
    writeJson(sessionStore, SESSION_KEY, session);
    notifySessionChange();
  }
}

export function getAdminSession() {
  const session = readJson(sessionStore, ADMIN_SESSION_KEY);
  if (!session) return null;
  // Expire admin sessions after 12 hours even within a long-running browser tab.
  const SESSION_TTL_MS = 12 * 60 * 60 * 1000;
  if (!session.savedAt || Date.now() - session.savedAt > SESSION_TTL_MS) {
    sessionStore?.removeItem(ADMIN_SESSION_KEY);
    return null;
  }
  return session;
}

export function createAdminSessionFromUser(session) {
  if (!session?.token || session.user?.role !== "admin") return null;
  const adminSession = {
    token: session.token,
    adminId: session.user.email,
    savedAt: Date.now(),
  };
  sessionStore?.removeItem(SESSION_KEY);
  writeJson(sessionStore, ADMIN_SESSION_KEY, adminSession);
  _cache.clear();
  notifySessionChange();
  return adminSession;
}

export function logoutUser() {
  sessionStore?.removeItem(SESSION_KEY);
  sessionStore?.removeItem(ADMIN_SESSION_KEY);
  legacyStore?.removeItem(SESSION_KEY);
  legacyStore?.removeItem(ADMIN_SESSION_KEY);
  _cache.clear();
  notifySessionChange();
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
        durationSec: assessment.recordings?.[qId]?.durationSeconds ?? null,
        audioFileId: assessment.audioFileIds?.[qId] || null,
      })),
      recordingCount: assessment.recordingCount || 0,
      skipBackgroundInference: Boolean(assessment.skipBackgroundInference),
    }),
  });

  // Return a shape compatible with what the frontend expects
  return {
    id: data.assessment.id,
    userId: data.assessment.userId,
    score: data.assessment.score,
    severity: data.assessment.severity,
    status: data.assessment.status,
    reportStatus: data.assessment.reportStatus,
    isReportReady: data.assessment.isReportReady,
    createdAt: data.assessment.createdAt,
    answers: assessment.answers,
    recordingCount: assessment.recordingCount,
    userName: assessment.userName,
    email: assessment.email,
    role: assessment.role,
  };
}

export async function scoreQuestionAudio({
  questionId,
  audioFileId,
  durationSec,
}) {
  return apiFetch("/assessments/score/question", {
    method: "POST",
    body: JSON.stringify({
      questionId: Number(questionId),
      audioFileId,
      durationSec: durationSec ?? null,
    }),
  });
}

export async function listAssessments() {
  const data = await apiFetch("/assessments?page=1&pageSize=100");
  const user = getCurrentUser();
  // Map to the shape the frontend expects
  return (data.items || []).map((a) => ({
    id: a.id,
    score: a.score,
    severity: a.severity,
    recordingCount: a.recordingCount,
    status: a.status,
    reportStatus: a.reportStatus,
    isReportReady: a.isReportReady,
    doctorRemarks: a.doctorRemarks,
    createdAt: a.createdAt,
    userId: user?.id,
    userName: user?.name,
    email: user?.email,
    role: user?.role,
    mlScore: a.mlScore,
    mlSeverity: a.mlSeverity,
  }));
}

export async function getLatestAssessment() {
  const data = await apiFetch("/assessments/latest");
  return data.assessment;
}

export async function getAssessmentDetail(assessmentId) {
  const data = await apiFetch(`/assessments/${assessmentId}`, {
    skipCache: true,
  });
  return data.assessment;
}

// ── Dashboard ───────────────────────────────────────────

export async function getDashboardSnapshot() {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = getAdminSession();

  if (adminSession?.token) {
    try {
      return await apiFetch("/admin/dashboard/snapshot");
    } catch {
      // fallback
    }
  }

  if (session?.user?.role === "doctor") {
    try {
      // Fetch all three in parallel for lower latency
      const [summary, trends, alerts] = await Promise.all([
        apiFetch("/doctor/dashboard/summary"),
        apiFetch("/doctor/dashboard/patient-trends?limit=50"),
        apiFetch("/doctor/dashboard/alerts?limit=12"),
      ]);

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
        patientCount: summary.patientCount ?? summary.totals?.patients ?? 0,
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

// ── Doctors ───────────────────────────────────────────

export async function listDoctors({ minFee, maxFee, isAvailable } = {}) {
  const params = new URLSearchParams();
  if (minFee !== "" && minFee != null) params.set("minFee", minFee);
  if (maxFee !== "" && maxFee != null) params.set("maxFee", maxFee);
  if (isAvailable !== "" && isAvailable != null) {
    params.set("isAvailable", String(isAvailable));
  }
  const query = params.toString();
  const data = await apiFetch(`/doctors${query ? `?${query}` : ""}`, {
    skipCache: true,
  });
  return data.items || [];
}

export async function getDoctorProfile() {
  const data = await apiFetch("/doctor/profile", { skipCache: true });
  return data.profile;
}

export async function updateDoctorProfile(profile) {
  const data = await apiFetch("/doctor/profile", {
    method: "PUT",
    body: JSON.stringify(profile),
  });
  return data.profile;
}

export async function getUserProfile() {
  const data = await apiFetch("/auth/me", { skipCache: true });
  return data.user;
}

export async function updateUserProfile(profile) {
  const data = await apiFetch("/auth/me", {
    method: "PUT",
    body: JSON.stringify(profile),
  });
  return data.user;
}

export async function assignDoctor({
  doctorId,
  assessmentId,
  autoAssign = false,
}) {
  const data = await apiFetch("/assign-doctor", {
    method: "POST",
    body: JSON.stringify({ doctorId, assessmentId, autoAssign }),
  });
  return data.assignment;
}

export async function listDoctorAssignments(status = "") {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const data = await apiFetch(`/doctor/assignments${query}`, {
    skipCache: true,
  });
  return data.items || [];
}

export async function updateDoctorAssignment(assignmentId, action) {
  const data = await apiFetch(`/doctor/assignments/${assignmentId}`, {
    method: "PATCH",
    body: JSON.stringify({ action }),
  });
  return data;
}

export async function getDoctorReport(assessmentId) {
  const data = await apiFetch(`/doctor/reports/${assessmentId}`, {
    skipCache: true,
  });
  return data;
}

export async function updateDoctorReportRemarks(assessmentId, doctorRemarks) {
  const data = await apiFetch(`/doctor/reports/${assessmentId}/remarks`, {
    method: "PUT",
    body: JSON.stringify({ doctorRemarks }),
  });
  return data.assessment;
}

export async function getDoctorPatientReports(patientId) {
  return apiFetch(`/reports/${patientId}`, {
    skipCache: true,
  });
}

export async function getDoctorPatientTrends(patientId) {
  return apiFetch(
    `/doctor/dashboard/patient-trends?patientId=${encodeURIComponent(patientId)}`,
    { skipCache: true },
  );
}

export async function listPatientAssignments() {
  const data = await apiFetch("/patient/assignments", { skipCache: true });
  return data.items || [];
}

// ── PHQ-8 Questions (from backend) ──────────────────────

export async function fetchQuestions() {
  return apiFetch("/phq8/questions");
}

// ── Audio Upload ───────────────────────────────────────

export async function uploadAudio(blob, filename = "recording.webm") {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
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

export async function getAudioBlobUrl(fileId) {
  const session = readJson(sessionStore, SESSION_KEY);
  const adminSession = readJson(sessionStore, ADMIN_SESSION_KEY);
  const token = session?.token || adminSession?.token;

  const res = await fetch(`${API_BASE}/files/audio/${fileId}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Audio fetch failed (${res.status})`);
  }

  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ── ML Details & Monitoring ────────────────────────────

export async function getProcessingStatus(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/processing-status`, {
    skipCache: true,
  });
}

export async function getMLDetails(assessmentId) {
  return apiFetch(`/assessments/${assessmentId}/ml-details`, {
    skipCache: true,
  });
}

export async function getAdminMetrics() {
  return apiFetch("/admin/dashboard/metrics");
}

export async function getMLHealth() {
  return apiFetch("/admin/dashboard/ml-health");
}
