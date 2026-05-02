import { useState, useRef, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import Card from '../components/Card.jsx';
import Button from '../components/Button.jsx';
import { verifyOtp, resendOtp } from '../services/api.js';

export default function VerifyOTP() {
  const navigate = useNavigate();
  const location = useLocation();
  const email = location.state?.email || '';

  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [loading, setLoading] = useState(false);
  const [resending, setResending] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [countdown, setCountdown] = useState(60);
  const [canResend, setCanResend] = useState(false);
  const inputRefs = useRef([]);

  // Redirect if no email in state
  useEffect(() => {
    if (!email) {
      navigate('/signup', { replace: true });
    }
  }, [email, navigate]);

  // Countdown timer
  useEffect(() => {
    if (countdown <= 0) {
      setCanResend(true);
      return;
    }
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  // Auto-focus first input
  useEffect(() => {
    if (inputRefs.current[0]) {
      inputRefs.current[0].focus();
    }
  }, []);

  const handleChange = (index, value) => {
    // Only allow digits
    if (value && !/^\d$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    setError('');

    // Auto-advance to next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // Auto-submit when all digits entered
    if (value && index === 5 && newOtp.every((d) => d !== '')) {
      handleVerify(newOtp.join(''));
    }
  };

  const handleKeyDown = (index, event) => {
    if (event.key === 'Backspace') {
      if (!otp[index] && index > 0) {
        // Move to previous input
        const newOtp = [...otp];
        newOtp[index - 1] = '';
        setOtp(newOtp);
        inputRefs.current[index - 1]?.focus();
      }
    } else if (event.key === 'ArrowLeft' && index > 0) {
      inputRefs.current[index - 1]?.focus();
    } else if (event.key === 'ArrowRight' && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handlePaste = (event) => {
    event.preventDefault();
    const text = event.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (text.length === 0) return;

    const newOtp = [...otp];
    for (let i = 0; i < 6; i++) {
      newOtp[i] = text[i] || '';
    }
    setOtp(newOtp);

    // Focus appropriate input
    const focusIndex = Math.min(text.length, 5);
    inputRefs.current[focusIndex]?.focus();

    // Auto-submit if all 6 digits
    if (text.length === 6) {
      handleVerify(text);
    }
  };

  const handleVerify = async (code) => {
    if (!code || code.length !== 6) {
      setError('Please enter the complete 6-digit code.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      await verifyOtp({ email, otp: code });
      setSuccess(true);
      // Wait for the success animation, then redirect
      setTimeout(() => navigate('/login', { replace: true }), 2200);
    } catch (err) {
      setError(err.message || 'Verification failed. Please try again.');
      setOtp(['', '', '', '', '', '']);
      inputRefs.current[0]?.focus();
    } finally {
      setLoading(false);
    }
  };

  const handleResend = async () => {
    if (!canResend) return;
    setResending(true);
    setError('');
    try {
      await resendOtp({ email });
      setCountdown(60);
      setCanResend(false);
    } catch (err) {
      setError(err.message || 'Failed to resend OTP.');
    } finally {
      setResending(false);
    }
  };

  const maskedEmail = email
    ? email.replace(/(.{2})(.*)(@.*)/, (_, a, b, c) => a + '*'.repeat(Math.min(b.length, 6)) + c)
    : '';

  if (success) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#F4FCF7] via-[#F7F7F2] to-[#EFF6F1] px-4">
        <div className="text-center animate-fade-in">
          {/* Animated checkmark */}
          <div className="relative w-32 h-32 mx-auto mb-8">
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] animate-pulse opacity-20" />
            <div className="absolute inset-2 rounded-full bg-gradient-to-br from-[#2D6A4F] to-[#52B788] flex items-center justify-center shadow-lg shadow-[#2D6A4F]/20">
              <svg className="w-16 h-16 text-white animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
              </svg>
            </div>
          </div>
          <h2 className="text-4xl font-bold text-[#1B3A2D] mb-3">Verified!</h2>
          <p className="text-lg text-[#66716B] mb-2">
            Your email has been verified successfully.
          </p>
          <p className="text-sm text-[#999]">Redirecting you to sign in...</p>

          {/* Animated progress bar */}
          <div className="mt-6 mx-auto w-48 h-1.5 bg-[#E8E8E8] rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-[#2D6A4F] to-[#52B788] rounded-full"
              style={{
                animation: 'progress 2s ease-out forwards',
              }}
            />
          </div>
          <style>{`
            @keyframes progress {
              from { width: 0%; }
              to { width: 100%; }
            }
          `}</style>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[#F4FCF7] via-[#F7F7F2] to-[#EFF6F1] px-4 py-10">
      <div className="w-full max-w-lg">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-br from-[#D8F3DC] to-[#B7E4C7] mb-6 shadow-lg shadow-[#2D6A4F]/10">
            <svg className="w-10 h-10 text-[#2D6A4F]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75" />
            </svg>
          </div>
          <p className="text-xs uppercase tracking-[0.22em] text-[#52B788] font-semibold mb-3">Email Verification</p>
          <h1 className="text-3xl lg:text-4xl font-bold text-[#1B1B1B] tracking-tight">Check your inbox</h1>
          <p className="mt-3 text-[#777] max-w-sm mx-auto">
            We&apos;ve sent a 6-digit verification code to{' '}
            <span className="font-semibold text-[#1B3A2D]">{maskedEmail}</span>
          </p>
        </div>

        {/* OTP Card */}
        <Card className="shadow-elevated p-8 md:p-10">
          <div className="space-y-8">
            {/* Error */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-center gap-3 animate-fade-in">
                <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                </svg>
                {error}
              </div>
            )}

            {/* OTP Input */}
            <div>
              <label className="block text-sm font-medium text-[#1B1B1B] mb-4 text-center">
                Enter verification code
              </label>
              <div className="flex justify-center gap-3" onPaste={handlePaste}>
                {otp.map((digit, index) => (
                  <input
                    key={index}
                    ref={(el) => { inputRefs.current[index] = el; }}
                    type="text"
                    inputMode="numeric"
                    maxLength={1}
                    value={digit}
                    onChange={(e) => handleChange(index, e.target.value)}
                    onKeyDown={(e) => handleKeyDown(index, e)}
                    className={`
                      w-14 h-16 text-center text-2xl font-bold rounded-xl border-2
                      transition-all duration-200 outline-none
                      ${digit
                        ? 'border-[#2D6A4F] bg-[#F4FCF7] text-[#1B3A2D] shadow-sm shadow-[#2D6A4F]/10'
                        : 'border-[#E8E8E8] bg-white text-[#1B1B1B] hover:border-[#B5B5B5]'
                      }
                      focus:border-[#2D6A4F] focus:ring-2 focus:ring-[#2D6A4F]/20 focus:bg-[#F4FCF7]
                    `}
                    disabled={loading}
                    id={`otp-input-${index}`}
                    aria-label={`Digit ${index + 1}`}
                  />
                ))}
              </div>
            </div>

            {/* Verify Button */}
            <Button
              type="button"
              fullWidth
              size="lg"
              disabled={loading || otp.some((d) => d === '')}
              onClick={() => handleVerify(otp.join(''))}
              className="mt-2"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Verifying...
                </span>
              ) : (
                'Verify Email'
              )}
            </Button>

            {/* Resend */}
            <div className="text-center">
              <p className="text-sm text-[#777]">
                Didn&apos;t receive the code?{' '}
                {canResend ? (
                  <button
                    type="button"
                    onClick={handleResend}
                    disabled={resending}
                    className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold transition-colors"
                  >
                    {resending ? 'Sending...' : 'Resend Code'}
                  </button>
                ) : (
                  <span className="text-[#999]">
                    Resend in{' '}
                    <span className="font-semibold text-[#2D6A4F] tabular-nums">
                      {Math.floor(countdown / 60)}:{String(countdown % 60).padStart(2, '0')}
                    </span>
                  </span>
                )}
              </p>
            </div>

            {/* Security note */}
            <div className="flex items-start gap-3 p-4 bg-[#F4FCF7] rounded-xl border border-[#D8F3DC]">
              <svg className="w-5 h-5 text-[#2D6A4F] flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
              <div>
                <p className="text-xs font-semibold text-[#2D6A4F]">Secure Verification</p>
                <p className="text-xs text-[#66716B] mt-0.5">
                  This code was sent from the MindScope admin account. Never share it with anyone.
                </p>
              </div>
            </div>
          </div>
        </Card>

        {/* Back to signup */}
        <p className="text-center mt-6 text-sm text-[#777]">
          Wrong email?{' '}
          <button
            type="button"
            onClick={() => navigate('/signup')}
            className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold"
          >
            Go back to Sign Up
          </button>
        </p>
      </div>
    </div>
  );
}
