import { useState } from 'react';
import { Link } from 'react-router-dom';
import { getCurrentUser } from '../services/api.js';

const faqData = [
  {
    question: 'How does the voice analysis work?',
    answer:
      'Our AI analyses vocal patterns, tone variations, and speech cadence from your recorded responses. Combined with the PHQ-8 questionnaire, it provides a comprehensive severity estimation.',
  },
  {
    question: 'What is the PHQ-8 screening assessment?',
    answer:
      'The PHQ-8 is a validated clinical tool that measures the severity of depressive symptoms over the past two weeks. It consists of eight targeted questions, each scored from 0 to 3, providing a total score that maps to a severity band.',
  },
  {
    question: 'How accurate is the severity estimation?',
    answer:
      'The estimation combines standardised clinical scoring with AI-driven voice analysis to provide a reliable indicator. It is designed as a screening tool, not a clinical diagnosis — always consult a healthcare professional.',
  },
  {
    question: 'Is my voice data kept private?',
    answer:
      'Absolutely. All recordings are processed securely and are never shared with third parties. You can review and delete your data at any time from your profile settings.',
  },
  {
    question: 'Can I retake the assessment?',
    answer:
      'Yes. You can retake the assessment as often as you like. Each session is saved independently so you can track your well-being over time.',
  },
];

export default function Landing() {
  const [openFaq, setOpenFaq] = useState(1); // second item open by default
  const currentUser = getCurrentUser();
  const isGuest = !currentUser;
  const isDoctor = currentUser?.role === 'doctor';
  const startLink = isGuest ? '/signup' : isDoctor ? '/doctor/dashboard' : '/assessment';
  const startLabel = isGuest ? 'Get Started Now' : isDoctor ? 'Open Doctor Dashboard' : 'Start Assessment';

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div className="pt-24 lg:pt-28">
      {/* ─── Hero Section ─── */}
      <section className="relative overflow-hidden">
        <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8 py-24 lg:py-36">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-16 items-center">
            {/* Left — Text */}
            <div className="space-y-10 animate-fade-in">
              <div className="landing-badge">
                PHQ-8 Check-in
              </div>
              <h1 className="landing-hero-title">
                Understand your mood with a focused
                <span className="landing-hero-accent"> screening flow</span>
              </h1>
              <p className="landing-hero-subtitle">
                Move through the eight PHQ-8 questions, record your answers, and review a clear severity score at the end.
              </p>
              <div className="flex flex-wrap gap-4">
                {isGuest ? (
                  <>
                    <Link to="/signup">
                      <button className="landing-btn-primary">
                        Create Account
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                      </button>
                    </Link>
                    <Link to="/login">
                      <button className="landing-btn-outline">
                        Sign In
                      </button>
                    </Link>
                  </>
                ) : (
                  <Link to={isDoctor ? '/doctor/dashboard' : '/assessment'}>
                    <button className="landing-btn-primary">
                      {isDoctor ? 'Open Doctor Dashboard' : 'Start Assessment'}
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                      </svg>
                    </button>
                  </Link>
                )}
                <a href="#how-it-works">
                  <button className="landing-btn-outline">
                    Learn More
                  </button>
                </a>
              </div>
            </div>

            {/* Right — Voice Waveform Illustration */}
            <div className="hidden lg:block animate-slide-up">
              <div className="relative">
                <div className="landing-hero-bg-decoration" />
                <div className="landing-hero-card">
                  <div className="flex flex-col items-center">
                    <div className="relative">
                      <div className="landing-mic-circle">
                        <svg className="w-10 h-10 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                        </svg>
                      </div>
                      <div className="absolute inset-0 rounded-full border-2 border-[#2D6A4F]/20 animate-ping" />
                      <div className="absolute -inset-3 rounded-full border border-[#2D6A4F]/10 animate-pulse" />
                    </div>
                    <p className="mt-4 text-sm font-semibold text-[#1B1B1B]">Speak Naturally</p>
                    <p className="text-xs text-[#777] mt-1">Record each answer with playback</p>
                  </div>

                  {/* Animated waveform bars */}
                  <div className="flex items-end justify-center gap-1 h-16">
                    {[40, 65, 45, 80, 55, 70, 35, 90, 60, 50, 75, 40, 85, 55, 45, 70, 50, 65, 40, 55].map((h, i) => (
                      <div
                        key={i}
                        className="w-1.5 rounded-full"
                        style={{
                          height: `${h}%`,
                          background: 'linear-gradient(to top, #2D6A4F, #52B788)',
                          animation: `waveform 1.5s ease-in-out ${i * 0.08}s infinite alternate`,
                        }}
                      />
                    ))}
                  </div>

                  {/* Feature highlights */}
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { label: '8 Questions', icon: '📝' },
                      { label: 'Voice Replay', icon: '▶️' },
                      { label: 'Severity Score', icon: '📊' },
                    ].map((item) => (
                      <div key={item.label} className="landing-feature-chip">
                        <p className="text-xl mb-1">{item.icon}</p>
                        <p className="text-[10px] font-medium text-[#555]">{item.label}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>


      {/* ─── How It Works ─── */}
      <section id="how-it-works" className="landing-section-white">
        <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <p className="landing-section-label">Process</p>
            <h2 className="landing-section-title">How It Works</h2>
            <p className="landing-section-subtitle">
              The flow is short, focused, and built around the PHQ-8 screening format.
            </p>
          </div>

          <div className="relative">
            <div className="hidden md:block absolute left-20 right-20 top-8 h-px bg-gradient-to-r from-transparent via-[#2D6A4F]/30 to-transparent" />
            <div className="grid md:grid-cols-3 gap-10 lg:gap-14">
              {[
                {
                  step: '01', title: 'Answer the Questions',
                  desc: 'Go through the PHQ-8 prompts one by one and select the answer that fits best.',
                  icon: (
                    <svg className="w-7 h-7 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.172a2 2 0 011.414.586l5.828 5.828A2 2 0 0120 10.828V19a2 2 0 01-2 2z" />
                    </svg>
                  ),
                },
                {
                  step: '02', title: 'Record Your Response',
                  desc: 'Use the voice recorder for each question and review the playback before moving on.',
                  icon: (
                    <svg className="w-7 h-7 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
                    </svg>
                  ),
                },
                {
                  step: '03', title: 'Review Your Score',
                  desc: 'See the cumulative PHQ-8 score, severity band, and a clear summary of your result.',
                  icon: (
                    <svg className="w-7 h-7 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h3m-3 4h6m-6-8h6m3 10.5V5.25A2.25 2.25 0 0015.75 3H8.25A2.25 2.25 0 006 5.25v15A2.25 2.25 0 008.25 22.5h7.5A2.25 2.25 0 0018 20.25z" />
                    </svg>
                  ),
                },
              ].map((item, i) => (
                <article key={i} className="text-center relative group px-3">
                  <div className="landing-step-icon">
                    {item.icon}
                  </div>
                  <p className="landing-step-label">Step {item.step}</p>
                  <h3 className="landing-step-title">{item.title}</h3>
                  <p className="landing-step-desc">{item.desc}</p>
                </article>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ─── FAQ Section ─── */}
      <section className="landing-section-white">
        <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-[1fr_1.4fr] gap-16 lg:gap-24 items-start">
            {/* Left — Heading */}
            <div>
              <p className="landing-section-label">FAQ</p>
              <h2 className="landing-faq-title">
                Frequently asked<br />questions
              </h2>
            </div>

            {/* Right — Accordion */}
            <div className="landing-faq-list">
              {faqData.map((item, index) => (
                <div key={index} className="landing-faq-item">
                  <button
                    onClick={() => toggleFaq(index)}
                    className="landing-faq-trigger"
                  >
                    <span className="landing-faq-question">{item.question}</span>
                    <span className="landing-faq-icon">
                      {openFaq === index ? '×' : '+'}
                    </span>
                  </button>
                  <div
                    className={`landing-faq-answer ${openFaq === index ? 'landing-faq-answer--open' : ''}`}
                  >
                    <p className="landing-faq-answer-text">{item.answer}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ─── CTA Banner ─── */}
      <section className="landing-cta-section">
        <div className="max-w-[88rem] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="landing-cta-card">
            <div className="landing-cta-content">
              <h2 className="landing-cta-title">
                Take the first step towards{' '}
                <em className="landing-cta-italic">understanding</em>
              </h2>
              <p className="landing-cta-subtitle">
                Join thousands of people who use MindScope<br />
                for fast and reliable mental health screening.
              </p>
              <Link to={startLink}>
                <button className="landing-cta-btn">{startLabel}</button>
              </Link>
            </div>
            {/* Decorative elements */}
            <div className="landing-cta-decoration">
              <div className="landing-cta-star" />
              <div className="landing-cta-star landing-cta-star--small" />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
