import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Card from "../components/Card.jsx";
import Input from "../components/Input.jsx";
import Button from "../components/Button.jsx";
import { forgotPassword, resetPassword } from "../services/api.js";

export default function ForgotPassword() {
  const navigate = useNavigate();
  const [step, setStep] = useState("email"); // email | otp | done
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const handleRequestOtp = async (e) => {
    e.preventDefault();
    const nextErrors = {};
    if (!email.trim()) nextErrors.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))
      nextErrors.email = "Invalid email format";
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors);
      return;
    }

    setLoading(true);
    setErrors({});
    try {
      const data = await forgotPassword({ email: email.trim() });
      setMessage(data.message);
      setStep("otp");
    } catch (err) {
      setErrors({ submit: err.message || "Failed to send OTP" });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async (e) => {
    e.preventDefault();
    const nextErrors = {};
    if (!otp.trim()) nextErrors.otp = "OTP is required";
    else if (otp.trim().length !== 6) nextErrors.otp = "OTP must be 6 digits";
    if (!newPassword) nextErrors.newPassword = "New password is required";
    else if (newPassword.length < 6)
      nextErrors.newPassword = "Password must be at least 6 characters";
    if (newPassword !== confirmPassword)
      nextErrors.confirmPassword = "Passwords do not match";
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors);
      return;
    }

    setLoading(true);
    setErrors({});
    try {
      await resetPassword({
        email: email.trim(),
        otp: otp.trim(),
        newPassword,
      });
      setStep("done");
    } catch (err) {
      setErrors({ submit: err.message || "Failed to reset password" });
    } finally {
      setLoading(false);
    }
  };

  const handleResendOtp = async () => {
    setLoading(true);
    try {
      await forgotPassword({ email: email.trim() });
      setMessage("A new OTP has been sent to your email.");
    } catch {
      setMessage("Failed to resend OTP. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#EFF9F2] via-[#F7F7F2] to-[#ECF3EE] flex items-center justify-center px-4 py-8">
      <div className="w-full max-w-md">
        {/* Logo / Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-[#D8F3DC] to-[#B7E4C7] mb-5 shadow-sm">
            <svg
              className="w-7 h-7 text-[#1B3A2D]"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 5.25a3 3 0 013 3m3 0a6 6 0 01-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1121.75 8.25z"
              />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-[#1B1B1B] mb-2">
            {step === "done" ? "Password Reset" : "Forgot Password"}
          </h2>
          <p className="text-[#66716B]">
            {step === "email" && "Enter your email and we'll send an OTP to reset your password."}
            {step === "otp" && "Enter the OTP sent to your email and choose a new password."}
            {step === "done" && "Your password has been changed successfully."}
          </p>
        </div>

        <div className="bg-white/70 backdrop-blur-xl rounded-2xl border border-[#E8E8E8]/60 shadow-[0_8px_32px_rgba(0,0,0,0.06)] p-8">
          {/* Step 1: Email */}
          {step === "email" && (
            <form onSubmit={handleRequestOtp} className="space-y-5">
              {errors.submit && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
                  {errors.submit}
                </div>
              )}

              <Input
                label="Email Address"
                id="email"
                type="email"
                placeholder="your@email.com"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  if (errors.email) setErrors((p) => ({ ...p, email: "" }));
                }}
                error={errors.email}
                required
              />

              <Button type="submit" fullWidth size="lg" disabled={loading}>
                {loading ? "Sending..." : "Send OTP"}
              </Button>
            </form>
          )}

          {/* Step 2: OTP + New Password */}
          {step === "otp" && (
            <form onSubmit={handleReset} className="space-y-5">
              {errors.submit && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm">
                  {errors.submit}
                </div>
              )}
              {message && (
                <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-700 text-sm">
                  {message}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-[#1B1B1B] mb-1.5">
                  Verification Code
                </label>
                <div className="flex justify-center gap-2">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <input
                      key={i}
                      type="text"
                      inputMode="numeric"
                      maxLength={1}
                      className="w-11 h-12 text-center text-lg font-bold border border-[#E8E8E8] rounded-xl bg-white focus:outline-none focus:ring-2 focus:ring-[#52B788]/30 focus:border-[#52B788] transition-all"
                      value={otp[i] || ""}
                      onChange={(e) => {
                        const val = e.target.value.replace(/\D/g, "");
                        if (!val && otp[i]) {
                          // Deleted
                          const next = otp.slice(0, i) + otp.slice(i + 1);
                          setOtp(next);
                          return;
                        }
                        if (!val) return;
                        const next = otp.slice(0, i) + val + otp.slice(i + 1);
                        setOtp(next.slice(0, 6));
                        if (val && i < 5) {
                          const nextInput = e.target.parentElement.children[i + 1];
                          nextInput?.focus();
                        }
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Backspace" && !otp[i] && i > 0) {
                          const prev = e.target.parentElement.children[i - 1];
                          prev?.focus();
                        }
                      }}
                    />
                  ))}
                </div>
                {errors.otp && (
                  <p className="text-xs text-red-500 mt-1">{errors.otp}</p>
                )}
              </div>

              <Input
                label="New Password"
                id="newPassword"
                type="password"
                placeholder="Enter new password"
                value={newPassword}
                onChange={(e) => {
                  setNewPassword(e.target.value);
                  if (errors.newPassword)
                    setErrors((p) => ({ ...p, newPassword: "" }));
                }}
                error={errors.newPassword}
                required
              />

              <Input
                label="Confirm Password"
                id="confirmPassword"
                type="password"
                placeholder="Confirm new password"
                value={confirmPassword}
                onChange={(e) => {
                  setConfirmPassword(e.target.value);
                  if (errors.confirmPassword)
                    setErrors((p) => ({ ...p, confirmPassword: "" }));
                }}
                error={errors.confirmPassword}
                required
              />

              <Button type="submit" fullWidth size="lg" disabled={loading}>
                {loading ? "Resetting..." : "Reset Password"}
              </Button>

              <button
                type="button"
                onClick={handleResendOtp}
                disabled={loading}
                className="w-full text-center text-sm text-[#2D6A4F] hover:text-[#1B3A2D] font-medium transition-colors disabled:opacity-50"
              >
                Didn't receive the code? Resend OTP
              </button>
            </form>
          )}

          {/* Step 3: Done */}
          {step === "done" && (
            <div className="text-center space-y-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-50 border-2 border-emerald-200">
                <svg
                  className="w-8 h-8 text-emerald-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
              <p className="text-sm text-[#555]">
                Your password has been reset successfully. You can now sign in with your new password.
              </p>
              <Button
                fullWidth
                size="lg"
                onClick={() => navigate("/login")}
              >
                Go to Sign In
              </Button>
            </div>
          )}

          {step !== "done" && (
            <p className="text-center text-sm text-[#66716B] mt-6">
              Remember your password?{" "}
              <Link
                to="/login"
                className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold transition-colors"
              >
                Sign In
              </Link>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
