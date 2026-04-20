import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Loader from '../components/Loader.jsx';
import { getProcessingStatus } from '../services/api.js';

const processingSteps = [
  {
    label: 'Uploading Audio',
    description: 'Sending voice recordings to the server',
    duration: 1200,
  },
  {
    label: 'Analyzing Voice Patterns',
    description: 'Running ML model on audio features',
    duration: 2500,
  },
  {
    label: 'Scoring Assessment',
    description: 'Computing PHQ-8 severity',
    duration: 1400,
  },
  {
    label: 'Opening Results',
    description: 'Moving to the score report',
    duration: 900,
  },
];

const fastSteps = [
  {
    label: 'Scoring Answers',
    description: 'Adding the selected PHQ-8 values',
    duration: 1200,
  },
  {
    label: 'Preparing Report',
    description: 'Formatting the score summary',
    duration: 1000,
  },
  {
    label: 'Opening Results',
    description: 'Moving to the score report',
    duration: 800,
  },
];

export default function Processing() {
  const [activeStep, setActiveStep] = useState(0);
  const navigate = useNavigate();
  const pollRef = useRef(null);

  // Determine if ML processing is happening
  const assessment = JSON.parse(sessionStorage.getItem('latestAssessment') || '{}');
  const hasAudio = (assessment.recordingCount || 0) > 0;
  const steps = hasAudio ? processingSteps : fastSteps;

  useEffect(() => {
    if (!hasAudio) {
      // Fast path: fake timer steps
      if (activeStep < steps.length) {
        const timer = setTimeout(() => setActiveStep((p) => p + 1), steps[activeStep].duration);
        return () => clearTimeout(timer);
      }
      const navTimer = setTimeout(() => navigate('/results'), 800);
      return () => clearTimeout(navTimer);
    }

    // ML path: poll backend for status
    if (activeStep === 0) {
      const t = setTimeout(() => setActiveStep(1), steps[0].duration);
      return () => clearTimeout(t);
    }

    if (activeStep === 1 && assessment.id) {
      // Poll every 2s until completed
      const poll = async () => {
        try {
          const data = await getProcessingStatus(assessment.id);
          if (data.status === 'completed') {
            setActiveStep(2);
          }
        } catch {
          // On error, just proceed
          setActiveStep(2);
        }
      };
      poll();
      pollRef.current = setInterval(poll, 2000);
      // Timeout after 30s regardless
      const timeout = setTimeout(() => setActiveStep(2), 30000);
      return () => {
        clearInterval(pollRef.current);
        clearTimeout(timeout);
      };
    }

    if (activeStep === 2) {
      const t = setTimeout(() => setActiveStep(3), steps[2].duration);
      return () => clearTimeout(t);
    }

    if (activeStep >= steps.length) {
      const navTimer = setTimeout(() => navigate('/results'), 800);
      return () => clearTimeout(navTimer);
    }
  }, [activeStep, hasAudio, navigate, assessment.id, steps]);

  const isComplete = activeStep >= steps.length;
  const totalSteps = steps.length;
  const progress = isComplete ? 100 : ((activeStep + 1) / totalSteps) * 100;

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-12 bg-[#F7F7F2]">
      <style>{`
        @keyframes shimmer {
          0%, 100% { opacity: 0.25; transform: translateY(0); }
          50% { opacity: 1; transform: translateY(-2px); }
        }
        @keyframes pulse-ring-anim {
          0% { transform: scale(1); opacity: 1; }
          100% { transform: scale(1.4); opacity: 0; }
        }
        @keyframes slide-in {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-shimmer { animation: shimmer 2s infinite; }
        .animate-pulse-ring-anim { animation: pulse-ring-anim 2s infinite; }
        .animate-slide-in { animation: slide-in 0.6s ease-out; }
      `}</style>

      <div className="w-full max-w-4xl text-center bg-white/80 backdrop-blur-md border border-[#E8E8E8] rounded-3xl shadow-[0_20px_60px_rgba(45,106,79,0.08)] px-6 py-10 md:px-10">
        <div className="flex justify-center mb-10 relative h-32">
          {!isComplete ? (
            <Loader size="lg" text={`Preparing results... ${Math.round(progress)}%`} />
          ) : (
            <div className="w-24 h-24 rounded-full bg-[#D8F3DC] flex items-center justify-center border-2 border-[#B7E4C7] animate-slide-in">
              <svg className="w-12 h-12 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          )}
        </div>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-[#1B1B1B] mb-2 animate-slide-in">
            {isComplete ? 'Complete!' : 'Preparing your result...'}
          </h1>
          <p className="text-base text-[#777] animate-slide-in" style={{ animationDelay: '0.1s' }}>
            {isComplete ? 'Your score report is ready' : 'Your PHQ-8 score is being formatted'}
          </p>
        </div>

        <div className="mb-8 px-2">
          <div className="h-2.5 bg-[#D8F3DC] rounded-full overflow-hidden border border-[#B7E4C7]">
            <div
              className="h-full bg-gradient-to-r from-[#52B788] to-[#2D6A4F] rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-[#B5B5B5] mt-2 font-medium">
            {Math.min(activeStep + 1, totalSteps)} of {totalSteps} steps
          </p>
        </div>

        <div className="space-y-3 mb-6">
          {steps.map((step, i) => {
            const isDone = i < activeStep;
            const isActive = i === activeStep;

            return (
              <div
                key={i}
                className={`relative flex items-center gap-3 p-3 rounded-lg border transition-all duration-300 ${isDone
                    ? 'bg-[#F0FAF4] border-[#B7E4C7]'
                    : isActive
                      ? 'bg-[#FAFAF7] border-[#2D6A4F]/40 shadow-sm scale-[1.02]'
                      : 'bg-white/60 border-[#E8E8E8] opacity-70'
                  }`}
                style={isActive ? { animation: 'slide-in 0.3s ease-out' } : {}}
              >
                <div className="flex-shrink-0 relative w-8 h-8">
                  {isDone ? (
                    <div className="w-full h-full rounded-full bg-[#D8F3DC] flex items-center justify-center border border-[#52B788]">
                      <svg className="w-4 h-4 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  ) : isActive ? (
                    <>
                      <div className="absolute inset-0 rounded-full border-2 border-[#52B788]/30 animate-pulse-ring-anim" style={{ animation: 'pulse-ring-anim 1.5s infinite' }} />
                      <div className="w-full h-full rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] flex items-center justify-center text-white text-sm font-semibold">
                        {i + 1}
                      </div>
                    </>
                  ) : (
                    <div className="w-full h-full rounded-full bg-gray-100 flex items-center justify-center border border-gray-200">
                      <div className="w-2 h-2 rounded-full bg-gray-400" />
                    </div>
                  )}
                </div>

                <div className="text-left flex-1">
                  <p className={`text-sm font-semibold transition-colors ${isDone ? 'text-[#2D6A4F]' : isActive ? 'text-[#1B1B1B]' : 'text-[#B5B5B5]'}`}>
                    {step.label}
                  </p>
                  <p className={`text-xs transition-colors ${isDone ? 'text-[#52B788]' : isActive ? 'text-[#777]' : 'text-[#D1D5DB]'}`}>
                    {step.description}
                  </p>
                </div>

                {isActive && (
                  <div className="flex gap-1">
                    {[0, 1, 2].map((j) => (
                      <div
                        key={j}
                        className="w-1.5 h-1.5 rounded-full bg-[#2D6A4F] animate-shimmer"
                        style={{ animationDelay: `${j * 0.2}s` }}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {!isComplete && (
          <p className="text-xs text-[#777] font-medium tracking-wide">
            Please keep this window open.
          </p>
        )}
      </div>
    </div>
  );
}
