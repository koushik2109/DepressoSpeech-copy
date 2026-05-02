import { useState, useEffect } from "react";
import { Link, Navigate } from "react-router-dom";
import { getCurrentUser, listAssessments } from "../services/api.js";

const severityTone = {
  Severe: "bg-red-50 text-red-700 border-red-200",
  "Moderately Severe": "bg-orange-50 text-orange-700 border-orange-200",
  Moderate: "bg-amber-50 text-amber-700 border-amber-200",
  Mild: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Minimal: "bg-green-50 text-green-700 border-green-200",
};

export default function AssessmentHistory() {
  const user = getCurrentUser();
  const [assessments, setAssessments] = useState([]);

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
  }, [user?.email, user?.id]);

  if (!user || user.role !== "patient") {
    return <Navigate to="/signin" replace />;
  }

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
                Track how your PHQ-8 scores changed over time. Tap any session
                for detailed ML metrics.
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
                const isCompleted =
                  item.status === "completed" ||
                  item.reportStatus === "available" ||
                  item.isReportReady;
                return (
                  <article
                    key={item.id}
                    className="rounded-xl border border-[#E8E8E8] bg-[#FAFAF7] px-4 py-4 transition-colors hover:border-[#B7E4C7] hover:bg-[#F3FBF7] md:px-5 md:py-5"
                  >
                    <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div>
                        <p className="text-sm text-[#6A766F]">
                          Session #{assessments.length - index}
                        </p>
                        <p className="text-xs text-[#9AA49F]">
                          {new Date(item.createdAt).toLocaleString()}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="rounded-full bg-[#ECF8F3] px-3 py-1 text-xs font-semibold text-[#1F7A66]">
                          Score {item.score}/24
                        </span>

                        {!isCompleted && item.status === "failed" && (
                          <span className="rounded-full bg-[#FEE2E2] px-3 py-1 text-xs font-semibold text-[#991B1B]">
                            Failed
                          </span>
                        )}
                        {!isCompleted && item.status !== "failed" && (
                          <span className="rounded-full bg-[#FEF3C7] px-3 py-1 text-xs font-semibold text-[#92400E]">
                            Processing
                          </span>
                        )}
                        {isCompleted && (
                          <span className="rounded-full bg-[#D8F3DC] px-3 py-1 text-xs font-semibold text-[#2D6A4F]">
                            Completed
                          </span>
                        )}
                        <span
                          className={`rounded-full border px-3 py-1 text-xs font-semibold ${severityTone[item.severity] || "bg-gray-50 text-gray-700 border-gray-200"}`}
                        >
                          {item.severity}
                        </span>
                        {isCompleted ? (
                          <Link
                            to={`/assessment-history/${item.id}`}
                            className="rounded-lg bg-[#1B3A2D] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#2D6A4F]"
                          >
                            Open Report
                          </Link>
                        ) : (
                          <span className="rounded-lg bg-[#E8E8E8] px-3 py-2 text-sm font-semibold text-[#9AA49F]">
                            {item.status === "failed" ? "Unavailable" : "Processing"}
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="mt-3 text-sm text-[#5F6B65]">
                      Responses captured: {item.recordingCount || 0} / 8
                      {item.mlSeverity && item.mlSeverity !== item.severity && (
                        <span className="ml-3 text-xs text-[#9AA49F]">
                          ML severity: {item.mlSeverity}
                        </span>
                      )}
                    </p>
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
