import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/Card.jsx";
import Button from "../components/Button.jsx";
import VoiceRecorder from "../components/VoiceRecorder.jsx";
import { buildQuestionSet, getSeverityLabel } from "../data/questionsData.js";
import {
  getCurrentUser,
  saveAssessment,
  uploadAudio,
} from "../services/api.js";

export default function Assessment() {
  const navigate = useNavigate();
  const [currentQ, setCurrentQ] = useState(0);
  const [voiceScores, setVoiceScores] = useState({});
  const [recordings, setRecordings] = useState({});

  const [submitting, setSubmitting] = useState(false);

  const user = useMemo(() => getCurrentUser(), []);
  const questions = useMemo(() => buildQuestionSet(), []);
  const question = questions[currentQ];

  const score = Object.values(voiceScores).reduce(
    (total, value) => total + Number(value || 0),
    0,
  );
  const progress = ((currentQ + 1) / questions.length) * 100;
  const isLast = currentQ === questions.length - 1;
  const hasRecording = Boolean(recordings[question?.id]);
  const completedCount = questions.filter((item) => recordings[item.id]).length;
  const upcomingQuestion = !isLast ? questions[currentQ + 1] : null;

  const handleRecordingComplete = (blob, previewUrl, durationSeconds) => {
    setRecordings((previous) => ({
      ...previous,
      [question.id]: {
        blob,
        previewUrl,
        durationSeconds,
      },
    }));

    // Score is determined by the ML model, NOT by recording duration.
    // Initialize with 0 — the real score comes back from the server
    // via ml_score after background inference completes.
    setVoiceScores((previous) => ({
      ...previous,
      [question.id]: 0,
    }));
  };

  const handleRecordingCleared = () => {
    setRecordings((previous) => {
      const next = { ...previous };
      delete next[question.id];
      return next;
    });

    setVoiceScores((previous) => {
      const next = { ...previous };
      delete next[question.id];
      return next;
    });
  };

  const handleNext = async () => {
    if (!hasRecording) {
      return;
    }

    if (isLast) {
      setSubmitting(true);
      try {
        // Upload all audio recordings IN PARALLEL for lower total latency
        const uploadEntries = Object.entries(recordings).filter(
          ([, rec]) => rec.blob,
        );
        const uploadResults = await Promise.allSettled(
          uploadEntries.map(([qId, rec]) =>
            uploadAudio(rec.blob, `q${qId}.webm`).then((res) => ({
              qId,
              fileId: res.fileId,
            })),
          ),
        );
        const audioFileIds = {};
        for (const result of uploadResults) {
          if (result.status === "fulfilled") {
            audioFileIds[result.value.qId] = result.value.fileId;
          } else {
            console.warn("Audio upload failed:", result.reason);
          }
        }

        const latestAssessment = {
          userId: user?.id || null,
          userName: user?.name || "",
          email: user?.email || "",
          role: user?.role || "",
          answers: voiceScores,
          audioFileIds,
          score,
          severity: getSeverityLabel(score),
          recordingCount: Object.keys(recordings).length,
          createdAt: new Date().toISOString(),
        };

        const saved = await saveAssessment(latestAssessment);
        sessionStorage.setItem("latestAssessment", JSON.stringify(saved));
        navigate("/processing");
      } catch (error) {
        console.error("Failed to save assessment:", error);
      } finally {
        setSubmitting(false);
      }
      return;
    }

    setCurrentQ((previous) => previous + 1);
  };

  const handlePrev = () => {
    if (currentQ > 0) {
      setCurrentQ((previous) => previous - 1);
    }
  };

  return (
    <div className="pt-24 lg:pt-28 min-h-screen px-4 py-12 bg-[#F7F7F2]">
      <div className="w-full max-w-[90rem] mx-auto animate-fade-in">
        <div className="text-center mb-10">
          <p className="text-xs tracking-[0.18em] uppercase font-semibold text-[#52B788] mb-3">
            PHQ-8 Assessment
          </p>
          <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
            {user?.name ? `Welcome back, ${user.name}` : "PHQ-8 Screening"}
          </h1>
          <p className="mt-3 text-base text-[#777]">
            Each question is on its own step. Record your response and continue.
          </p>
        </div>

        <div className="mb-8 max-w-4xl mx-auto">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-[#777] font-medium">
              Question {currentQ + 1} of {questions.length}
            </span>
            <span className="font-semibold text-[#2D6A4F]">
              Score {score}/24
            </span>
          </div>
          <div className="w-full h-2.5 bg-[#D8F3DC] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#52B788] to-[#2D6A4F] rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <Card className="shadow-elevated p-8 md:p-10 max-w-4xl mx-auto">
          <div className="space-y-8">
            <div className="text-left space-y-4">
              <div className="inline-flex items-center px-4 py-1.5 rounded-full bg-[#D8F3DC] text-[#2D6A4F] text-xs font-semibold tracking-wide uppercase">
                Question {currentQ + 1}
              </div>
              <h2 className="text-3xl font-semibold text-[#1B1B1B] leading-snug max-w-3xl">
                {question.text}
              </h2>
            </div>

            <div className="rounded-2xl border border-[#E8E8E8] bg-[#FAFAF7] p-5">
              <p className="text-sm text-[#555] leading-relaxed">
                Use the live waveform to keep a steady voice response. Record
                once, replay, and continue when ready.
              </p>
            </div>

            <div className="rounded-2xl border border-[#DDEBE2] bg-white p-5 md:p-6">
              <p className="text-xs tracking-[0.16em] uppercase font-semibold text-[#52B788] mb-2">
                Voice Recorder
              </p>
              <p className="text-sm text-[#777] mb-6">
                Completed {completedCount} of {questions.length} responses.
              </p>

              <VoiceRecorder
                key={question.id}
                onRecordingComplete={handleRecordingComplete}
                onRecordingCleared={handleRecordingCleared}
              />
            </div>

            {upcomingQuestion && hasRecording && (
              <div className="rounded-xl border border-[#E8E8E8] bg-[#F8FBF9] px-4 py-4">
                <p className="text-xs uppercase tracking-[0.16em] font-semibold text-[#52B788] mb-2">
                  Up Next
                </p>
                <p className="text-sm text-[#4A5550]">
                  {upcomingQuestion.text}
                </p>
              </div>
            )}

            <div className="flex items-center justify-between gap-4">
              <Button
                variant="ghost"
                onClick={handlePrev}
                disabled={currentQ === 0}
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M11 17l-5-5m0 0l5-5m-5 5h12"
                  />
                </svg>
                Previous
              </Button>

              <Button
                variant="primary"
                onClick={handleNext}
                disabled={!hasRecording || submitting}
              >
                {submitting
                  ? "Uploading..."
                  : isLast
                    ? "Submit Assessment"
                    : "Next Question"}
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M13 7l5 5m0 0l-5 5m5-5H6"
                  />
                </svg>
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
