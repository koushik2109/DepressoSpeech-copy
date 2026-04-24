import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import Card from "../components/Card.jsx";
import Input from "../components/Input.jsx";
import Button from "../components/Button.jsx";
import { registerUser, googleLogin } from "../services/api.js";

const roleCards = [
  {
    id: "patient",
    title: "Patient",
    description:
      "Take the PHQ-8 screening and keep your account for future sessions.",
    icon: (
      <svg
        className="w-8 h-8 text-[#2D6A4F]"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.964 0a9 9 0 10-11.964 0m11.964 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
        />
      </svg>
    ),
  },
  {
    id: "doctor",
    title: "Doctor",
    description:
      "Review assessments and monitor patient activity from the clinical side.",
    icon: (
      <svg
        className="w-8 h-8 text-[#2D6A4F]"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5m4.75-11.396a24.301 24.301 0 014.5 0m0 0v5.714a2.25 2.25 0 00.659 1.591L19 14.5m-9.25-11.396c-.251.023-.501.05-.75.082m7.5-.082c.251.023.501.05.75.082M19 14.5l-2.47 2.47a2.25 2.25 0 01-1.591.659H9.061a2.25 2.25 0 01-1.591-.659L5 14.5"
        />
      </svg>
    ),
  },
];

export default function SignUp() {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [role, setRole] = useState("");
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [form, setForm] = useState({
    name: "",
    age: "",
    email: "",
    password: "",
    confirmPassword: "",
    basicInfo: "",
    specialization: "",
    licenseNumber: "",
    clinicName: "",
    yearsExperience: "",
  });

  const handleGoogleResponse = useCallback(
    async (response) => {
      setLoading(true);
      setErrors({});
      try {
        const session = await googleLogin(response.credential);
        navigate(
          session.user.role === "doctor" ? "/doctor/dashboard" : "/assessment",
        );
      } catch (error) {
        setErrors({ submit: error.message || "Google sign-up failed" });
      } finally {
        setLoading(false);
      }
    },
    [navigate],
  );

  useEffect(() => {
    const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
    if (clientId && window.google?.accounts) {
      window.google.accounts.id.initialize({
        client_id: clientId,
        callback: handleGoogleResponse,
      });
      window.google.accounts.id.renderButton(
        document.getElementById("google-signup-btn"),
        {
          theme: "outline",
          size: "large",
          width: "100%",
          shape: "rectangular",
          text: "signup_with",
        },
      );
    }
  }, [handleGoogleResponse]);

  const handleChange = (field) => (event) => {
    setForm((previous) => ({ ...previous, [field]: event.target.value }));
    if (errors[field]) {
      setErrors((previous) => ({ ...previous, [field]: "" }));
    }
  };

  const validate = () => {
    const nextErrors = {};

    if (!role) nextErrors.role = "Choose a role to continue.";
    if (!form.name.trim()) nextErrors.name = "Name is required.";
    if (role === "patient" && (!form.age || Number(form.age) < 13))
      nextErrors.age = "Enter a valid age.";
    if (!form.email.trim()) nextErrors.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      nextErrors.email = "Enter a valid email.";
    if (!form.password) nextErrors.password = "Password is required.";
    else if (form.password.length < 8)
      nextErrors.password = "Password must be at least 8 characters.";
    if (form.password !== form.confirmPassword)
      nextErrors.confirmPassword = "Passwords do not match.";

    if (role === "patient" && !form.basicInfo.trim()) {
      nextErrors.basicInfo = "Add a short description about yourself.";
    }

    if (role === "doctor") {
      if (!form.specialization.trim())
        nextErrors.specialization = "Specialization is required.";
      if (!form.licenseNumber.trim())
        nextErrors.licenseNumber = "Medical license number is required.";
      if (!form.clinicName.trim())
        nextErrors.clinicName = "Hospital or clinic name is required.";
      if (!form.yearsExperience || Number(form.yearsExperience) < 0)
        nextErrors.yearsExperience = "Enter valid years of experience.";
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) return;

    setLoading(true);
    try {
      await registerUser({
        role,
        name: form.name.trim(),
        age: role === "patient" ? Number(form.age) : null,
        email: form.email.trim().toLowerCase(),
        password: form.password,
        basicInfo: role === "patient" ? form.basicInfo.trim() : "",
        specialization: role === "doctor" ? form.specialization.trim() : "",
        licenseNumber: role === "doctor" ? form.licenseNumber.trim() : "",
        clinicName: role === "doctor" ? form.clinicName.trim() : "",
        yearsExperience:
          role === "doctor" ? Number(form.yearsExperience) : null,
      });
      navigate("/verify-otp", {
        state: {
          email: form.email.trim().toLowerCase(),
          userName: form.name.trim(),
        },
      });
    } catch (error) {
      setErrors({ submit: error.message || "Unable to create account." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen px-4 py-10 bg-gradient-to-br from-[#F4FCF7] via-[#F7F7F2] to-[#EFF6F1]">
      <div className="w-full max-w-5xl mx-auto">
        {/* Stepper + Header */}
        <div className="mb-12">
          {/* Stepper */}
          <div className="flex items-center justify-center mb-10">
            <div className="flex items-center gap-0">
              {/* Step 1 */}
              <div className="flex flex-col items-center relative">
                <div
                  className={`w-11 h-11 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                    step >= 2
                      ? "bg-[#2D6A4F] text-white shadow-lg shadow-[#2D6A4F]/25"
                      : step === 1
                        ? "bg-[#2D6A4F] text-white shadow-lg shadow-[#2D6A4F]/25 ring-4 ring-[#D8F3DC]"
                        : "bg-[#E8E8E8] text-[#999]"
                  }`}
                >
                  {step >= 2 ? (
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M4.5 12.75l6 6 9-13.5"
                      />
                    </svg>
                  ) : (
                    "1"
                  )}
                </div>
                <span
                  className={`mt-2.5 text-xs font-semibold tracking-wide uppercase transition-colors ${step >= 1 ? "text-[#2D6A4F]" : "text-[#B5B5B5]"}`}
                >
                  Choose Role
                </span>
              </div>

              {/* Connector */}
              <div className="w-24 sm:w-32 h-[2px] mx-2 mt-[-18px] relative overflow-hidden rounded-full bg-[#E8E8E8]">
                <div
                  className={`absolute inset-y-0 left-0 bg-gradient-to-r from-[#2D6A4F] to-[#52B788] rounded-full transition-all duration-500 ease-out ${
                    step >= 2 ? "w-full" : "w-0"
                  }`}
                />
              </div>

              {/* Step 2 */}
              <div className="flex flex-col items-center relative">
                <div
                  className={`w-11 h-11 rounded-full flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                    step >= 2
                      ? "bg-[#2D6A4F] text-white shadow-lg shadow-[#2D6A4F]/25 ring-4 ring-[#D8F3DC]"
                      : "bg-[#F0F0F0] text-[#B5B5B5] border-2 border-[#E8E8E8]"
                  }`}
                >
                  2
                </div>
                <span
                  className={`mt-2.5 text-xs font-semibold tracking-wide uppercase transition-colors ${step >= 2 ? "text-[#2D6A4F]" : "text-[#B5B5B5]"}`}
                >
                  Your Details
                </span>
              </div>
            </div>
          </div>

          {/* Header text */}
          <div className="text-center">
            <h1 className="text-4xl lg:text-5xl font-bold text-[#1B1B1B] tracking-tight">
              {step === 1
                ? "How would you like to use MindScope?"
                : role === "doctor"
                  ? "Set up your doctor profile"
                  : "Tell us a bit about yourself"}
            </h1>
            <p className="mt-4 text-lg text-[#777] max-w-2xl mx-auto">
              {step === 1
                ? "Select your role below. This determines your dashboard experience and the features available to you."
                : "Fill in the details below to complete your registration. All fields marked with * are required."}
            </p>
          </div>
        </div>

        {step === 1 && (
          <>
            <div className="grid md:grid-cols-2 gap-6">
              {roleCards.map((card) => (
                <Card
                  key={card.id}
                  hover
                  selected={role === card.id}
                  onClick={() => {
                    setRole(card.id);
                    setErrors((previous) => ({ ...previous, role: "" }));
                    setStep(2);
                  }}
                  className="group cursor-pointer"
                >
                  <div className="flex flex-col items-center text-center gap-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#D8F3DC] to-[#B7E4C7] flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                      {card.icon}
                    </div>
                    <div>
                      <h2 className="text-2xl font-semibold text-[#1B1B1B] mb-2">
                        {card.title}
                      </h2>
                      <p className="text-sm text-[#777] max-w-sm mx-auto">
                        {card.description}
                      </p>
                    </div>
                    <div className="inline-flex items-center px-5 py-2.5 rounded-xl bg-gradient-to-r from-[#D8F3DC] to-[#B7E4C7] text-[#2D6A4F] text-sm font-semibold shadow-sm group-hover:shadow-md transition-shadow">
                      Continue
                      <svg
                        className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3"
                        />
                      </svg>
                    </div>
                  </div>
                </Card>
              ))}
              {errors.role && (
                <p className="md:col-span-2 text-sm text-red-500 text-center">
                  {errors.role}
                </p>
              )}
            </div>

            {/* Google Sign-Up divider */}
            <div className="max-w-md mx-auto mt-10">
              <div className="flex items-center gap-4 mb-6">
                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[#E8E8E8] to-transparent" />
                <span className="text-xs font-medium text-[#B5B5B5] uppercase tracking-wider">
                  or sign up with
                </span>
                <div className="flex-1 h-px bg-gradient-to-r from-transparent via-[#E8E8E8] to-transparent" />
              </div>
              <div id="google-signup-btn" className="flex justify-center" />
            </div>

            <p className="text-center text-sm text-[#777] mt-6">
              Already have an account?{" "}
              <button
                type="button"
                onClick={() => navigate("/login")}
                className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold transition-colors"
              >
                Sign In
              </button>
            </p>
          </>
        )}

        {step === 2 && (
          <div className="w-full max-w-2xl mx-auto">
            <div className="text-center mb-8">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="text-[#2D6A4F] hover:text-[#1B3A2D] font-medium mb-4 flex items-center gap-2 mx-auto transition-colors"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
                Back
              </button>
              <h2 className="text-3xl font-bold text-[#1B1B1B]">
                {role === "doctor" ? "Doctor Sign Up" : "Patient Sign Up"}
              </h2>
              <p className="mt-2 text-[#777]">
                Add your details to create the account.
              </p>
            </div>

            <div className="bg-white/70 backdrop-blur-xl rounded-2xl border border-[#E8E8E8]/60 shadow-[0_8px_32px_rgba(0,0,0,0.06)] p-8 md:p-10">
              <form className="space-y-6" onSubmit={handleSubmit}>
                {errors.submit && (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-sm flex items-start gap-2">
                    <svg
                      className="w-5 h-5 text-red-400 shrink-0 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
                      />
                    </svg>
                    {errors.submit}
                  </div>
                )}

                <Input
                  label="Name"
                  id="name"
                  placeholder="Jane Doe"
                  value={form.name}
                  onChange={handleChange("name")}
                  error={errors.name}
                  required
                />

                {role === "patient" ? (
                  <div className="grid md:grid-cols-2 gap-6">
                    <Input
                      label="Age"
                      id="age"
                      type="number"
                      placeholder="28"
                      value={form.age}
                      onChange={handleChange("age")}
                      error={errors.age}
                      required
                    />
                    <Input
                      label="Email"
                      id="email"
                      type="email"
                      placeholder="jane@example.com"
                      value={form.email}
                      onChange={handleChange("email")}
                      error={errors.email}
                      required
                    />
                  </div>
                ) : (
                  <div className="grid md:grid-cols-2 gap-6">
                    <Input
                      label="Specialization"
                      id="specialization"
                      placeholder="Psychiatry"
                      value={form.specialization}
                      onChange={handleChange("specialization")}
                      error={errors.specialization}
                      required
                    />
                    <Input
                      label="License Number"
                      id="licenseNumber"
                      placeholder="MED-548721"
                      value={form.licenseNumber}
                      onChange={handleChange("licenseNumber")}
                      error={errors.licenseNumber}
                      required
                    />
                  </div>
                )}

                {role === "doctor" && (
                  <div className="grid md:grid-cols-2 gap-6">
                    <Input
                      label="Hospital / Clinic"
                      id="clinicName"
                      placeholder="City Wellness Hospital"
                      value={form.clinicName}
                      onChange={handleChange("clinicName")}
                      error={errors.clinicName}
                      required
                    />
                    <Input
                      label="Years of Experience"
                      id="yearsExperience"
                      type="number"
                      placeholder="8"
                      value={form.yearsExperience}
                      onChange={handleChange("yearsExperience")}
                      error={errors.yearsExperience}
                      required
                    />
                  </div>
                )}

                {role === "doctor" && (
                  <Input
                    label="Email"
                    id="email"
                    type="email"
                    placeholder="doctor@example.com"
                    value={form.email}
                    onChange={handleChange("email")}
                    error={errors.email}
                    required
                  />
                )}

                {role === "patient" && (
                  <Input
                    label="Basic Info"
                    id="basicInfo"
                    placeholder="A short note about your background or what brings you here"
                    value={form.basicInfo}
                    onChange={handleChange("basicInfo")}
                    error={errors.basicInfo}
                    required
                  />
                )}

                <div className="grid md:grid-cols-2 gap-6">
                  <Input
                    label="Password"
                    id="password"
                    type="password"
                    placeholder="Create a password"
                    value={form.password}
                    onChange={handleChange("password")}
                    error={errors.password}
                    required
                  />
                  <Input
                    label="Confirm Password"
                    id="confirmPassword"
                    type="password"
                    placeholder="Repeat the password"
                    value={form.confirmPassword}
                    onChange={handleChange("confirmPassword")}
                    error={errors.confirmPassword}
                    required
                  />
                </div>

                <Button
                  type="submit"
                  fullWidth
                  size="lg"
                  disabled={loading}
                  className="mt-4"
                >
                  {loading ? "Creating Account..." : "Create Account"}
                </Button>

                <p className="text-center text-sm text-[#777]">
                  Already have an account?{" "}
                  <button
                    type="button"
                    onClick={() => navigate("/login")}
                    className="text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold transition-colors"
                  >
                    Sign In
                  </button>
                </p>
              </form>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
