import { useState, useEffect } from "react";
import { Link, Navigate } from "react-router-dom";
import { getCurrentUser, listAssessments, getMLDetails } from "../services/api.js";

const severityTone = {
  Severe: "bg-red-50 text-red-700 border-red-200",
  "Moderately Severe": "bg-orange-50 text-orange-700 border-orange-200",
  Moderate: "bg-amber-50 text-amber-700 border-amber-200",
  Mild: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Minimal: "bg-green-50 text-green-700 border-green-200",
};

function MetricRow({ label, value }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-[#F0F0EC] last:border-0">
      <span className="text-xs text-[#6A766F]">{label}</span>
      <span className="text-xs font-semibold text-[#1B1B1B]">{value}</span>
    </div>
  );
}

function MetricsPanel({ assessmentId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getMLDetails(assessmentId)
      .then((res) => setData(res.mlDetails ?? null))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [assessmentId]);

  if (loading) {
    return (
      <div className="mt-4 rounded-xl bg-[#F3FBF7] border border-[#D6E3DA] p-4 text-xs text-[#52B788] animate-pulse">
        Loading metrics…
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mt-4 rounded-xl bg-[#FFF9F0] border border-[#FFE5B4] p-4 text-xs text-[#B07D2C]">
        ML analysis not available for this session (audio may still be processing or was not submitted).
      </div>
    );
  }

  const behavioral = data.behavioral || {};

  return (
    <div className="mt-4 rounded-xl border border-[#D6E3DA] bg-[#F3FBF7] p-4 space-y-4">
      {/* Confidence */}
      <div>
        <p className="text-xs uppercase tracking-[0.16em] font-semibold text-[#52B788] mb-2">Prediction Confidence</p>
        <div className="bg-white rounded-lg p-3 border border-[#E8E8E8]">
          <MetricRow label="Confidence Mean" value={data.confidenceMean != null ? `${(data.confidenceMean * 100).toFixed(1)}%` : null} />
          <MetricRow label="Confidence Std Dev" value={data.confidenceStd != null ? `±${(data.confidenceStd * 100).toFixed(1)}%` : null} />
          <MetricRow
            label="95% CI"
            value={
              data.ciLower != null && data.ciUpper != null
                ? `${data.ciLower.toFixed(2)} – ${data.ciUpper.toFixed(2)}`
                : null
            }
          />
          <MetricRow label="Inference Time" value={data.inferenceTimeMs != null ? `${data.inferenceTimeMs.toFixed(0)} ms` : null} />
        </div>
      </div>

      {/* Audio Quality */}
      <div>
        <p className="text-xs uppercase tracking-[0.16em] font-semibold text-[#52B788] mb-2">Audio Quality</p>
        <div className="bg-white rounded-lg p-3 border border-[#E8E8E8]">
          <MetricRow label="Quality Score" value={data.audioQualityScore != null ? `${(data.audioQualityScore * 100).toFixed(1)}%` : null} />
          <MetricRow label="SNR" value={data.audioSnrDb != null ? `${data.audioSnrDb.toFixed(1)} dB` : null} />
          <MetricRow label="Speech Probability" value={data.audioSpeechProb != null ? `${(data.audioSpeechProb * 100).toFixed(1)}%` : null} />
        </div>
      </div>

      {/* Behavioral */}
      {Object.keys(behavioral).length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-[0.16em] font-semibold text-[#52B788] mb-2">Behavioral Signals</p>
          <div className="bg-white rounded-lg p-3 border border-[#E8E8E8]">
            {Object.entries(behavioral).map(([key, val]) => (
              <MetricRow
                key={key}
                label={key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                value={typeof val === "number" ? val.toFixed(3) : String(val)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function AssessmentHistory() {
  const user = getCurrentUser();
  const [assessments, setAssessments] = useState([]);
  const [expandedId, setExpandedId] = useState(null);

  useEffect(() => {
    listAssessments()
      .then((all) => {
        const filtered = all
          .filter((item) => {
            const sameUser =
              item.userId && user?.id ? item.userId === user.id : false;
            const sameEmail =
              item.email && user?.email
                ? item.email.toLowerCase() === user.email.toLowerCase()
                : false;
            return sameUser || sameEmail;
          })
          .sort(
            (a, b) =>
              new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
          );
        setAssessments(filtered);
      })
      .catch(() => setAssessments([]));
  }, []);

  if (!user || user.role !== "patient") {
    return <Navigate to="/signin" replace />;
  }

  const toggleExpand = (id) => setExpandedId((prev) => (prev === id ? null : id));

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-10 bg-[#F7F7F2]">
      <div className="w-full max-w-[88rem] mx-auto space-y-8">
        <section className="rounded-3xl border border-[#D6E3DA] bg-gradient-to-br from-[#F3FBF7] via-white to-[#EEF7F2] px-6 py-8 md:px-10">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] font-semibold text-[#52B788]">
                Assessment History
              </p>
              <h1 className="mt-3 text-3xl md:text-4xl font-bold text-[#1B1B1B]">
                Your Past Assessments
              </h1>
              <p className="mt-2 text-[#5F6B65]">
                Track how your PHQ-8 scores changed over time. Tap any session for detailed ML metrics.
              </p>
            </div>
            <div className="flex flex-wrap gap-3">
              <Link
                to="/assessment"
                className="inline-flex items-center rounded-xl bg-[#1B3A2D] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#2D6A4F] transition-colors"
              >
                Start Assessment
              </Link>
              <Link
                to="/"
                className="inline-flex items-center rounded-xl border border-[#D6E3DA] bg-white px-5 py-2.5 text-sm font-semibold text-[#1B1B1B] hover:bg-[#F4FAF6] transition-colors"
              >
                Back Home
              </Link>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-[#E8E8E8] bg-white p-6 md:p-8">
          {assessments.length === 0 ? (
            <div className="text-center py-14">
              <p className="text-xl font-semibold text-[#1B1B1B] mb-2">
                No assessments yet
              </p>
              <p className="text-sm text-[#6A766F] mb-6">
                Complete your first PHQ-8 assessment to start building history.
              </p>
              <Link
                to="/assessment"
                className="inline-flex items-center rounded-xl bg-[#1B3A2D] text-white px-5 py-2.5 text-sm font-semibold hover:bg-[#2D6A4F] transition-colors"
              >
                Start First Assessment
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {assessments.map((item, index) => {
                const isExpanded = expandedId === item.id;
                return (
                  <article
                    key={item.id}
                    className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-4 md:px-5 md:py-5"
                  >
                    <button
                      type="button"
                      onClick={() => toggleExpand(item.id)}
                      className="w-full text-left"
                    >
                      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                        <div>
                          <p className="text-sm text-[#6A766F]">
                            Session #{assessments.length - index}
                          </p>
                          <p className="text-xs text-[#9AA49F]">
                            {new Date(item.createdAt).toLocaleString()}
                          </p>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="rounded-full bg-[#ECF8F3] px-3 py-1 text-xs font-semibold text-[#1F7A66]">
                            Score {item.score}/24
                          </span>
                          {item.mlScore != null && (
                            <span className="rounded-full bg-[#EEF4FF] px-3 py-1 text-xs font-semibold text-[#3B5BDB]">
                              ML {item.mlScore.toFixed(1)}
                            </span>
                          )}
                          <span
                            className={`rounded-full border px-3 py-1 text-xs font-semibold ${severityTone[item.severity] || "bg-gray-50 text-gray-700 border-gray-200"}`}
                          >
                            {item.severity}
                          </span>
                          <span className="ml-1 text-[#9AA49F]">
                            <svg
                              className={`w-4 h-4 transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
                              fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                            </svg>
                          </span>
                        </div>
                      </div>
                      <p className="mt-3 text-sm text-[#5F6B65]">
                        Responses captured: {item.recordingCount || 0} / 8
                        {item.mlSeverity && item.mlSeverity !== item.severity && (
                          <span className="ml-3 text-xs text-[#9AA49F]">ML severity: {item.mlSeverity}</span>
                        )}
                      </p>
                    </button>

                    {isExpanded && <MetricsPanel assessmentId={item.id} />}
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
