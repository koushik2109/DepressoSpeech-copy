import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/Card.jsx";
import Button from "../components/Button.jsx";
import VoiceRecorder from "../components/VoiceRecorder.jsx";
import { buildQuestionSet, getSeverityLabel } from "../data/questionsData.js";
import {
  getCurrentUser,
  scoreQuestionAudio,
  saveAssessment,
  uploadAudio,
} from "../services/api.js";

function clampScore3(value) {
  return Math.max(0, Math.min(3, Math.round(Number(value || 0))));
}

export default function Assessment() {
  const navigate = useNavigate();
  const [currentQ, setCurrentQ] = useState(0);
  const [voiceScores, setVoiceScores] = useState({});
  const [recordings, setRecordings] = useState({});
  const [audioFileIds, setAudioFileIds] = useState({});
  const [scoringQuestionId, setScoringQuestionId] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [lastLatencyMs, setLastLatencyMs] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");

  const user = useMemo(() => getCurrentUser(), []);
  const questions = useMemo(() => buildQuestionSet(), []);
  const question = questions[currentQ];
  const questionId = question?.id;
  const hasRecording = Boolean(recordings[questionId]);
  const existingScore = voiceScores[questionId];
  const isLast = currentQ === questions.length - 1;
  const isScoringCurrent = scoringQuestionId === questionId;
  const isBusy = submitting || Boolean(scoringQuestionId);
  const canProceed = hasRecording && !isBusy;

  const score = Object.values(voiceScores).reduce(
    (total, value) => total + Number(value || 0),
    0,
  );
  const progress = ((currentQ + 1) / questions.length) * 100;
  const completedCount = questions.filter((item) => recordings[item.id]).length;
  const upcomingQuestion = !isLast ? questions[currentQ + 1] : null;

  const handleRecordingComplete = (blob, previewUrl, durationSeconds) => {
    setErrorMessage("");
    setRecordings((previous) => ({
      ...previous,
      [questionId]: {
        blob,
        previewUrl,
        durationSeconds,
      },
    }));

    // New recording invalidates existing score and uploaded file for this question.
    setVoiceScores((previous) => {
      const next = { ...previous };
      delete next[questionId];
      return next;
    });
    setAudioFileIds((previous) => {
      const next = { ...previous };
      delete next[questionId];
      return next;
    });
  };

  const handleRecordingCleared = () => {
    setErrorMessage("");
    setRecordings((previous) => {
      const next = { ...previous };
      delete next[questionId];
      return next;
    });
    setVoiceScores((previous) => {
      const next = { ...previous };
      delete next[questionId];
      return next;
    });
    setAudioFileIds((previous) => {
      const next = { ...previous };
      delete next[questionId];
      return next;
    });
  };

  const handleNext = async () => {
    if (!hasRecording || isBusy) {
      return;
    }
    setErrorMessage("");

    let questionScore = Number(existingScore);
    let currentAudioFileId = audioFileIds[questionId];

    if (!Number.isFinite(questionScore) || !currentAudioFileId) {
      setScoringQuestionId(questionId);
      try {
        const currentRecording = recordings[questionId];
        if (!currentRecording?.blob) {
          throw new Error("Recording is required for this question.");
        }

        const uploaded = await uploadAudio(currentRecording.blob, `q${questionId}.webm`);
        currentAudioFileId = uploaded.fileId;

        const scored = await scoreQuestionAudio({
          questionId,
          audioFileId: currentAudioFileId,
          durationSec: currentRecording.durationSeconds ?? null,
        });
        questionScore = clampScore3(scored.score);
        setLastLatencyMs(Number(scored.inferenceTimeMs ?? 0));
      } catch (error) {
        setErrorMessage(error.message || "Failed to score this question.");
        return;
      } finally {
        setScoringQuestionId(null);
      }
    }

    const nextScores = {
      ...voiceScores,
      [questionId]: clampScore3(questionScore),
    };
    const nextAudioFileIds = {
      ...audioFileIds,
      [questionId]: currentAudioFileId,
    };
    setVoiceScores(nextScores);
    setAudioFileIds(nextAudioFileIds);

    if (isLast) {
      setSubmitting(true);
      try {
        const finalScore = Object.values(nextScores).reduce(
          (total, value) => total + Number(value || 0),
          0,
        );

        const latestAssessment = {
          userId: user?.id || null,
          userName: user?.name || "",
          email: user?.email || "",
          role: user?.role || "",
          answers: nextScores,
          audioFileIds: nextAudioFileIds,
          recordings,
          score: finalScore,
          severity: getSeverityLabel(finalScore),
          recordingCount: Object.keys(recordings).length,
          skipBackgroundInference: false,
          createdAt: new Date().toISOString(),
        };

        const saved = await saveAssessment(latestAssessment);
        sessionStorage.setItem("latestAssessment", JSON.stringify(saved));
        navigate("/processing");
      } catch (error) {
        setErrorMessage(error.message || "Failed to save assessment.");
      } finally {
        setSubmitting(false);
      }
      return;
    }

    setCurrentQ((previous) => previous + 1);
  };

  const handlePrev = () => {
    if (currentQ > 0 && !isBusy) {
      setErrorMessage("");
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
            Record your answer, then continue. The model scores each answer out of 3.
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
                Click Next to run model scoring for this question from your recorded audio.
              </p>
            </div>

            <div className="rounded-2xl border border-[#DDEBE2] bg-white p-5 md:p-6">
              <p className="text-xs tracking-[0.16em] uppercase font-semibold text-[#52B788] mb-2">
                Voice Recorder
              </p>
              <p className="text-sm text-[#777] mb-2">
                Completed {completedCount} of {questions.length} responses.
              </p>
              {Number.isFinite(Number(existingScore)) ? (
                <p className="text-sm font-semibold text-[#2D6A4F] mb-4">
                  Current question score: {clampScore3(existingScore)}/3
                </p>
              ) : (
                <p className="text-sm text-[#6A766F] mb-4">
                  Current question score: pending
                </p>
              )}
              {lastLatencyMs != null && (
                <p className="text-xs text-[#6A766F] mb-4">
                  Last model response time: {(lastLatencyMs / 1000).toFixed(2)}s
                </p>
              )}

              <VoiceRecorder
                key={question.id}
                onRecordingComplete={handleRecordingComplete}
                onRecordingCleared={handleRecordingCleared}
              />
            </div>

            {errorMessage && (
              <div className="rounded-xl border border-[#F1C7C7] bg-[#FFF4F4] px-4 py-3">
                <p className="text-sm font-medium text-[#A94442]">{errorMessage}</p>
              </div>
            )}

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
                disabled={currentQ === 0 || isBusy}
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
                disabled={!canProceed}
              >
                {isScoringCurrent
                  ? "Scoring..."
                  : submitting
                    ? "Submitting..."
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
