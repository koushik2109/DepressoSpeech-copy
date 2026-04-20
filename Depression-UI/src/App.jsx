/**
 * App.jsx
 * Root component — defines all routes and wraps pages with Navbar.
 */
import { Suspense, lazy } from "react";
import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar.jsx";
import Loader from "./components/Loader.jsx";
import { getCurrentUser, getAdminSession } from "./services/api.js";

const Landing = lazy(() => import("./pages/Landing.jsx"));
const SignIn = lazy(() => import("./pages/SignIn.jsx"));
const SignUp = lazy(() => import("./pages/SignUp.jsx"));
const VerifyOTP = lazy(() => import("./pages/VerifyOTP.jsx"));
const ForgotPassword = lazy(() => import("./pages/ForgotPassword.jsx"));
const AdminLogin = lazy(() => import("./pages/AdminLogin.jsx"));
const AdminDashboard = lazy(() => import("./pages/AdminDashboard.jsx"));
const DoctorDashboard = lazy(() => import("./pages/DoctorDashboard.jsx"));
const Assessment = lazy(() => import("./pages/Assessment.jsx"));
const AssessmentHistory = lazy(() => import("./pages/AssessmentHistory.jsx"));
const Processing = lazy(() => import("./pages/Processing.jsx"));
const Results = lazy(() => import("./pages/Results.jsx"));

function App() {
  const location = useLocation();
  const currentUser = getCurrentUser();
  const authenticatedRedirect = "/";

  const hideNavbarOnRoutes = [
    "/login",
    "/signin",
    "/signup",
    "/admin",
    "/admin/dashboard",
    "/verify-otp",
    "/forgot-password",
  ];
  const shouldShowNavbar = !hideNavbarOnRoutes.includes(location.pathname);
  const canAccessPatientFlow = currentUser?.role === "patient";
  const canAccessDoctorFlow = currentUser?.role === "doctor";
  const adminSession = getAdminSession();

  return (
    <div className="min-h-screen bg-[#F7F7F2] font-inter">
      {shouldShowNavbar && <Navbar />}
      <main>
        <Suspense
          fallback={
            <div className="min-h-[60vh] flex items-center justify-center px-4">
              <Loader size="lg" text="Loading page..." />
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route
              path="/login"
              element={
                currentUser ? (
                  <Navigate to={authenticatedRedirect} replace />
                ) : (
                  <SignIn />
                )
              }
            />
            <Route path="/signin" element={<Navigate to="/login" replace />} />
            <Route
              path="/signup"
              element={
                currentUser ? (
                  <Navigate to={authenticatedRedirect} replace />
                ) : (
                  <SignUp />
                )
              }
            />
            <Route path="/verify-otp" element={<VerifyOTP />} />
            <Route path="/forgot-password" element={<ForgotPassword />} />
            <Route path="/admin" element={<AdminLogin />} />
            <Route
              path="/admin/dashboard"
              element={
                adminSession?.token ? (
                  <AdminDashboard />
                ) : (
                  <Navigate to="/" replace />
                )
              }
            />
            <Route
              path="/doctor/dashboard"
              element={
                canAccessDoctorFlow ? (
                  <DoctorDashboard />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/assessment"
              element={
                canAccessPatientFlow ? (
                  <Assessment />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/processing"
              element={
                canAccessPatientFlow ? (
                  <Processing />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/assessment-history"
              element={
                canAccessPatientFlow ? (
                  <AssessmentHistory />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
            <Route
              path="/results"
              element={
                canAccessPatientFlow ? (
                  <Results />
                ) : (
                  <Navigate to="/login" replace />
                )
              }
            />
          </Routes>
        </Suspense>
      </main>
    </div>
  );
}

export default App;
