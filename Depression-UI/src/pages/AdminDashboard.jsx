import { useState, useEffect } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { getDashboardSnapshot, getAdminSession } from "../services/api.js";
import MonitoringTab from "../components/MonitoringTab.jsx";

const TAB_ICONS = {
  overview:
    "M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z",
  users:
    "M15 19.128a9.38 9.38 0 002.625.372 9.337 9.337 0 004.121-.952 4.125 4.125 0 00-7.533-2.493M15 19.128v-.003c0-1.113-.285-2.16-.786-3.07M15 19.128v.106A12.318 12.318 0 018.624 21c-2.331 0-4.512-.645-6.374-1.766l-.001-.109a6.375 6.375 0 0111.964-3.07M12 6.375a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zm8.25 2.25a2.625 2.625 0 11-5.25 0 2.625 2.625 0 015.25 0z",
  assessments:
    "M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25z",
  settings:
    "M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.324.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 011.37.49l1.296 2.247a1.125 1.125 0 01-.26 1.431l-1.003.827c-.293.24-.438.613-.431.992a6.759 6.759 0 010 .255c-.007.378.138.75.43.99l1.005.828c.424.35.534.954.26 1.43l-1.298 2.247a1.125 1.125 0 01-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.57 6.57 0 01-.22.128c-.331.183-.581.495-.644.869l-.213 1.28c-.09.543-.56.941-1.11.941h-2.594c-.55 0-1.02-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 01-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 01-1.369-.49l-1.297-2.247a1.125 1.125 0 01.26-1.431l1.004-.827c.292-.24.437-.613.43-.992a6.932 6.932 0 010-.255c.007-.378-.138-.75-.43-.99l-1.004-.828a1.125 1.125 0 01-.26-1.43l1.297-2.247a1.125 1.125 0 011.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.087.22-.128.332-.183.582-.495.644-.869l.214-1.281z M15 12a3 3 0 11-6 0 3 3 0 016 0z",
  monitoring:
    "M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6",
};

/* ── Theme tokens ── */
const themes = {
  light: {
    bg: "bg-[#F4F6F9]",
    sidebarBg: "rgba(255,255,255,0.92)",
    sidebarBorder: "border-gray-200/70",
    headerBg: "rgba(255,255,255,0.85)",
    headerBorder: "border-gray-200/60",
    cardBg: "rgba(255,255,255,0.85)",
    cardBorder: "border-gray-200/60",
    cardHover: "hover:bg-gray-50/60",
    tableRowBorder: "border-gray-100",
    tableRowHover: "hover:bg-gray-50/70",
    tableHeadBg: "bg-gray-50/80",
    text1: "text-[#111827]",
    text2: "text-[#374151]",
    text3: "text-[#6B7280]",
    text4: "text-[#9CA3AF]",
    accent: "text-[#059669]",
    accentBg: "bg-[#059669]",
    accentLight: "bg-emerald-50 text-emerald-700",
    inputBg:
      "bg-white border-gray-200 text-[#111827] placeholder:text-[#9CA3AF]",
    inputFocus: "focus:ring-emerald-500/20 focus:border-emerald-500",
    filterBg: "bg-gray-100",
    filterActive: "bg-white text-[#111827] shadow-sm",
    filterInactive: "text-[#6B7280] hover:text-[#374151]",
    badgeDoctor: "bg-emerald-50 text-emerald-700 border-emerald-200",
    badgePatient: "bg-violet-50 text-violet-700 border-violet-200",
    sevSevere: "bg-red-50 text-red-700 border-red-200",
    sevModerate: "bg-amber-50 text-amber-700 border-amber-200",
    sevMild: "bg-yellow-50 text-yellow-700 border-yellow-200",
    sevNone: "bg-emerald-50 text-emerald-700 border-emerald-200",
    signOut: "text-red-600 hover:bg-red-50 border-red-200/60",
    gradientHeading: "from-[#111827] to-[#374151]",
    statGradients: {
      blue: "from-blue-50 to-blue-100/50 border-blue-200/60",
      emerald: "from-emerald-50 to-emerald-100/50 border-emerald-200/60",
      violet: "from-violet-50 to-violet-100/50 border-violet-200/60",
      amber: "from-amber-50 to-amber-100/50 border-amber-200/60",
    },
    statIcon: {
      blue: "text-blue-600",
      emerald: "text-emerald-600",
      violet: "text-violet-600",
      amber: "text-amber-600",
    },
    navActive: "bg-emerald-50 text-emerald-700",
    navInactive: "text-[#6B7280] hover:text-[#374151] hover:bg-gray-100/70",
    adminBoxBg: "bg-gray-50 border-gray-200/70",
    glowOrbs: false,
    overlayBg: "bg-black/30",
    dateBg: "bg-gray-100 text-[#6B7280] border-gray-200",
    toggleBg: "bg-gray-200",
    toggleKnob: "bg-white",
    svcRowBg: "bg-gray-50/80 border-gray-100",
    insightValue: "from-emerald-600 to-cyan-600",
    settingSectionBg: "bg-white border-gray-200/70",
    settingSectionHover: "hover:shadow-md hover:shadow-gray-200/50",
    themeCardBorder: "border-gray-200",
    themeCardActiveBorder: "border-emerald-500 ring-2 ring-emerald-500/20",
    themeCardBg: "bg-white",
    switchTrackOff: "bg-gray-300",
    switchTrackOn: "bg-emerald-500",
    switchKnob: "bg-white",
  },
  dark: {
    bg: "bg-[#0B0F19]",
    sidebarBg: "rgba(13, 17, 28, 0.95)",
    sidebarBorder: "border-white/[0.06]",
    headerBg: "rgba(11, 15, 25, 0.8)",
    headerBorder: "border-white/[0.06]",
    cardBg: "rgba(15, 23, 42, 0.5)",
    cardBorder: "border-white/[0.06]",
    cardHover: "hover:bg-white/[0.03]",
    tableRowBorder: "border-white/[0.03]",
    tableRowHover: "hover:bg-white/[0.02]",
    tableHeadBg: "",
    text1: "text-[#E2E8F0]",
    text2: "text-[#CBD5E1]",
    text3: "text-[#64748B]",
    text4: "text-[#475569]",
    accent: "text-emerald-400",
    accentBg: "bg-emerald-500",
    accentLight: "bg-emerald-500/15 text-emerald-400",
    inputBg:
      "bg-white/[0.04] border-white/[0.08] text-[#E2E8F0] placeholder:text-[#475569]",
    inputFocus: "focus:ring-emerald-500/20 focus:border-emerald-500/30",
    filterBg: "bg-white/[0.04] border border-white/[0.08]",
    filterActive: "bg-emerald-500/20 text-emerald-400 shadow-sm",
    filterInactive: "text-[#64748B] hover:text-[#94A3B8]",
    badgeDoctor: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    badgePatient: "bg-violet-500/10 text-violet-400 border-violet-500/20",
    sevSevere: "bg-red-500/10 text-red-400 border-red-500/20",
    sevModerate: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    sevMild: "bg-yellow-500/10 text-yellow-400 border-yellow-500/20",
    sevNone: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    signOut: "text-red-400 hover:bg-red-500/10 border-red-500/10",
    gradientHeading: "from-white to-[#94A3B8]",
    statGradients: {
      blue: "from-blue-500/20 to-blue-600/5 border-blue-500/10",
      emerald: "from-emerald-500/20 to-emerald-600/5 border-emerald-500/10",
      violet: "from-violet-500/20 to-violet-600/5 border-violet-500/10",
      amber: "from-amber-500/20 to-amber-600/5 border-amber-500/10",
    },
    statIcon: {
      blue: "text-blue-400",
      emerald: "text-emerald-400",
      violet: "text-violet-400",
      amber: "text-amber-400",
    },
    navActive:
      "bg-gradient-to-r from-emerald-500/15 to-cyan-500/10 text-emerald-400 shadow-sm shadow-emerald-500/5",
    navInactive: "text-[#64748B] hover:text-[#94A3B8] hover:bg-white/[0.03]",
    adminBoxBg: "bg-white/[0.03] border-white/[0.06]",
    glowOrbs: true,
    overlayBg: "bg-black/60",
    dateBg: "bg-white/[0.04] text-[#475569] border-white/[0.06]",
    toggleBg: "bg-white/10",
    toggleKnob: "bg-white",
    svcRowBg: "bg-white/[0.02] border-white/[0.04]",
    insightValue: "from-emerald-400 to-cyan-400",
    settingSectionBg: "bg-white/[0.03] border-white/[0.06]",
    settingSectionHover: "hover:shadow-none",
    themeCardBorder: "border-white/[0.08]",
    themeCardActiveBorder: "border-emerald-500 ring-2 ring-emerald-500/20",
    themeCardBg: "bg-white/[0.04]",
    switchTrackOff: "bg-white/20",
    switchTrackOn: "bg-emerald-500",
    switchKnob: "bg-white",
  },
};

const THEME_KEY = "mindscope-admin-theme";

export default function AdminDashboard() {
  const navigate = useNavigate();
  const session = getAdminSession();
  const [snapshot, setSnapshot] = useState({
    totals: { users: 0, doctors: 0, patients: 0, assessments: 0 },
    users: [],
    assessments: [],
  });
  const [activeTab, setActiveTab] = useState("overview");
  const [roleFilter, setRoleFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState(
    () => localStorage.getItem(THEME_KEY) || "light",
  );

  const t = themes[theme];

  useEffect(() => {
    getDashboardSnapshot()
      .then((data) => data && setSnapshot(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  if (!session?.token) {
    return <Navigate to="/" replace />;
  }

  const handleSignOut = () => {
    localStorage.removeItem("mindscope-admin-session");
    navigate("/");
  };

  const filteredUsers = snapshot.users.filter((u) => {
    if (roleFilter !== "all" && u.role !== roleFilter) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        u.name?.toLowerCase().includes(q) || u.email?.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "users", label: "Users" },
    { id: "assessments", label: "Assessments" },
    { id: "monitoring", label: "Monitoring" },
    { id: "settings", label: "Settings" },
  ];

  const isDark = theme === "dark";

  return (
    <div
      className={`min-h-screen ${t.bg} ${t.text1} flex transition-colors duration-300`}
    >
      {/* Ambient glow (dark only) */}
      {t.glowOrbs && (
        <div className="fixed inset-0 pointer-events-none overflow-hidden">
          <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-emerald-600/[0.07] blur-[120px]" />
          <div className="absolute -bottom-40 -right-40 w-[500px] h-[500px] rounded-full bg-cyan-600/[0.05] blur-[120px]" />
        </div>
      )}

      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        >
          <div className={`absolute inset-0 ${t.overlayBg} backdrop-blur-sm`} />
        </div>
      )}

      {/* ───── SIDEBAR ───── */}
      <aside
        className={`fixed lg:sticky top-0 left-0 z-50 lg:z-30 h-screen w-64 flex-shrink-0 flex flex-col border-r ${t.sidebarBorder} transition-transform duration-300 lg:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
        style={{ background: t.sidebarBg, backdropFilter: "blur(20px)" }}
      >
        {/* Logo */}
        <div
          className={`flex items-center gap-3 px-5 h-16 border-b ${t.sidebarBorder}`}
        >
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <svg
              className="w-5 h-5 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
              />
            </svg>
          </div>
          <div>
            <p className={`text-sm font-bold tracking-tight ${t.text1}`}>
              MindScope
            </p>
            <p
              className={`text-[10px] ${t.accent} font-semibold tracking-widest uppercase`}
            >
              Admin Panel
            </p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className={`ml-auto lg:hidden p-1 rounded-lg ${t.cardHover}`}
          >
            <svg
              className="w-5 h-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id);
                setSidebarOpen(false);
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all duration-200 ${
                activeTab === tab.id ? t.navActive : t.navInactive
              }`}
            >
              <svg
                className="w-[18px] h-[18px] flex-shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d={TAB_ICONS[tab.id]}
                />
              </svg>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-3 pb-4 space-y-2">
          <div className={`p-3 rounded-xl border ${t.adminBoxBg}`}>
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-600 flex items-center justify-center text-[11px] font-bold text-white">
                A
              </div>
              <div className="flex-1 min-w-0">
                <p className={`text-xs font-medium ${t.text1} truncate`}>
                  {session.adminId}
                </p>
                <p className={`text-[10px] ${t.text4}`}>Super Admin</p>
              </div>
            </div>
          </div>
          <button
            onClick={handleSignOut}
            className={`w-full flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-medium border transition-all ${t.signOut}`}
          >
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={1.5}
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15m3 0l3-3m0 0l-3-3m3 3H9"
              />
            </svg>
            Sign Out
          </button>
        </div>
      </aside>

      {/* ───── MAIN ───── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header
          className={`sticky top-0 z-20 h-14 flex items-center justify-between gap-4 px-4 sm:px-6 border-b ${t.headerBorder}`}
          style={{ background: t.headerBg, backdropFilter: "blur(16px)" }}
        >
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(true)}
              className={`lg:hidden p-1.5 rounded-lg ${t.cardHover} transition-colors`}
            >
              <svg
                className={`w-5 h-5 ${t.text3}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
                />
              </svg>
            </button>
            <h2 className={`text-sm font-semibold ${t.text1} capitalize`}>
              {activeTab === "overview" ? "Dashboard" : activeTab}
            </h2>
          </div>
          <div className="flex items-center gap-3">
            {/* Theme toggle */}
            <button
              onClick={() => setTheme(isDark ? "light" : "dark")}
              className={`p-2 rounded-xl transition-all ${t.cardHover} border ${isDark ? "border-white/[0.06]" : "border-gray-200"}`}
              title={isDark ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDark ? (
                <svg
                  className="w-4 h-4 text-amber-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M12 3v2.25m6.364.386l-1.591 1.591M21 12h-2.25m-.386 6.364l-1.591-1.591M12 18.75V21m-4.773-4.227l-1.591 1.591M5.25 12H3m4.227-4.773L5.636 5.636M15.75 12a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0z"
                  />
                </svg>
              ) : (
                <svg
                  className="w-4 h-4 text-[#6B7280]"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M21.752 15.002A9.718 9.718 0 0118 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 003 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 009.002-5.998z"
                  />
                </svg>
              )}
            </button>
            <span
              className={`hidden sm:block text-[11px] font-mono px-2.5 py-1 rounded-lg border ${t.dateBg}`}
            >
              {new Date().toLocaleDateString("en-US", {
                weekday: "short",
                month: "short",
                day: "numeric",
              })}
            </span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-4 sm:p-6 overflow-auto">
          {/* ========== OVERVIEW ========== */}
          {activeTab === "overview" && (
            <div className="space-y-6 animate-fadeIn">
              <div>
                <h1
                  className={`text-2xl font-bold bg-gradient-to-r ${t.gradientHeading} bg-clip-text text-transparent`}
                >
                  Welcome back, Admin
                </h1>
                <p className={`text-sm ${t.text3} mt-1`}>
                  Here's what's happening on your platform
                </p>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                  {
                    label: "Total Users",
                    value: snapshot.totals.users,
                    color: "blue",
                    icon: TAB_ICONS.users,
                  },
                  {
                    label: "Doctors",
                    value: snapshot.totals.doctors,
                    color: "emerald",
                    icon: "M4.26 10.147a60.438 60.438 0 00-.491 6.347A48.62 48.62 0 0112 20.904a48.62 48.62 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.636 50.636 0 00-2.658-.813A59.906 59.906 0 0112 3.493a59.903 59.903 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.717 50.717 0 0112 13.489a50.702 50.702 0 017.74-3.342",
                  },
                  {
                    label: "Patients",
                    value: snapshot.totals.patients,
                    color: "violet",
                    icon: "M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z",
                  },
                  {
                    label: "Assessments",
                    value: snapshot.totals.assessments,
                    color: "amber",
                    icon: TAB_ICONS.assessments,
                  },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className={`relative overflow-hidden rounded-2xl border bg-gradient-to-br ${t.statGradients[stat.color]} p-5 transition-shadow hover:shadow-lg`}
                    style={isDark ? { backdropFilter: "blur(12px)" } : {}}
                  >
                    <div className="relative">
                      <div className="flex items-center justify-between mb-3">
                        <span
                          className={`text-xs font-medium ${t.text3} uppercase tracking-wider`}
                        >
                          {stat.label}
                        </span>
                        <svg
                          className={`w-5 h-5 ${t.statIcon[stat.color]} opacity-60`}
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={1.5}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d={stat.icon}
                          />
                        </svg>
                      </div>
                      <p
                        className={`text-3xl font-bold tracking-tight ${t.text1}`}
                      >
                        {stat.value}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid xl:grid-cols-2 gap-5">
                <GlassCard
                  t={t}
                  isDark={isDark}
                  title="Recent Users"
                  action={() => setActiveTab("users")}
                  actionLabel="View all"
                >
                  {snapshot.users.length === 0 ? (
                    <EmptyState t={t} text="No users yet" />
                  ) : (
                    <div className="space-y-1">
                      {snapshot.users.slice(0, 6).map((user) => (
                        <div
                          key={user.id}
                          className={`flex items-center justify-between py-2.5 px-3 rounded-xl ${t.cardHover} transition-colors`}
                        >
                          <div className="flex items-center gap-3">
                            <Avatar name={user.name} role={user.role} />
                            <div>
                              <p className={`text-sm font-medium ${t.text1}`}>
                                {user.name}
                              </p>
                              <p className={`text-[11px] ${t.text4}`}>
                                {user.email}
                              </p>
                            </div>
                          </div>
                          <RoleBadge role={user.role} t={t} />
                        </div>
                      ))}
                    </div>
                  )}
                </GlassCard>

                <GlassCard
                  t={t}
                  isDark={isDark}
                  title="Recent Assessments"
                  action={() => setActiveTab("assessments")}
                  actionLabel="View all"
                >
                  {snapshot.assessments.length === 0 ? (
                    <EmptyState t={t} text="No assessments yet" />
                  ) : (
                    <div className="space-y-1">
                      {snapshot.assessments.slice(0, 6).map((a) => (
                        <div
                          key={a.id}
                          className={`flex items-center justify-between py-2.5 px-3 rounded-xl ${t.cardHover} transition-colors`}
                        >
                          <div>
                            <p className={`text-sm font-medium ${t.text1}`}>
                              {a.userName || "Anonymous"}
                            </p>
                            <p className={`text-[11px] ${t.text4}`}>
                              {a.createdAt
                                ? new Date(a.createdAt).toLocaleDateString()
                                : ""}
                            </p>
                          </div>
                          <SeverityBadge
                            severity={a.severity}
                            score={a.score}
                            t={t}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </GlassCard>
              </div>
            </div>
          )}

          {/* ========== USERS ========== */}
          {activeTab === "users" && (
            <div className="space-y-5 animate-fadeIn">
              <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
                <div>
                  <h1
                    className={`text-2xl font-bold bg-gradient-to-r ${t.gradientHeading} bg-clip-text text-transparent`}
                  >
                    User Management
                  </h1>
                  <p className={`text-sm ${t.text3} mt-1`}>
                    {filteredUsers.length} registered user
                    {filteredUsers.length !== 1 ? "s" : ""}
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-wrap">
                  <div className="relative">
                    <svg
                      className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${t.text4}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
                      />
                    </svg>
                    <input
                      type="text"
                      placeholder="Search users..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className={`pl-9 pr-4 py-2 text-sm rounded-xl border ${t.inputBg} focus:outline-none focus:ring-2 ${t.inputFocus} w-52 transition-all`}
                    />
                  </div>
                  <div className={`flex rounded-xl ${t.filterBg} p-0.5`}>
                    {["all", "doctor", "patient"].map((r) => (
                      <button
                        key={r}
                        onClick={() => setRoleFilter(r)}
                        className={`px-3.5 py-1.5 text-xs font-medium rounded-[10px] transition-all capitalize ${roleFilter === r ? t.filterActive : t.filterInactive}`}
                      >
                        {r === "all" ? "All" : r + "s"}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div
                className={`rounded-2xl border ${t.cardBorder} overflow-hidden`}
                style={
                  isDark
                    ? { background: t.cardBg, backdropFilter: "blur(12px)" }
                    : { background: t.cardBg }
                }
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr
                        className={`border-b ${isDark ? "border-white/[0.06]" : "border-gray-200"} ${t.tableHeadBg}`}
                      >
                        {["User", "Role", "Age", "Details", "Joined"].map(
                          (h) => (
                            <th
                              key={h}
                              className={`text-left px-5 py-3.5 text-[11px] font-semibold ${t.text3} uppercase tracking-widest`}
                            >
                              {h}
                            </th>
                          ),
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredUsers.length === 0 ? (
                        <tr>
                          <td colSpan={5}>
                            <EmptyState t={t} text="No users found" />
                          </td>
                        </tr>
                      ) : (
                        filteredUsers.map((user) => (
                          <tr
                            key={user.id}
                            className={`border-b ${t.tableRowBorder} ${t.tableRowHover} transition-colors`}
                          >
                            <td className="px-5 py-3.5">
                              <div className="flex items-center gap-3">
                                <Avatar name={user.name} role={user.role} />
                                <div>
                                  <p className={`font-medium ${t.text1}`}>
                                    {user.name}
                                  </p>
                                  <p className={`text-[11px] ${t.text4}`}>
                                    {user.email}
                                  </p>
                                </div>
                              </div>
                            </td>
                            <td className="px-5 py-3.5">
                              <RoleBadge role={user.role} t={t} />
                            </td>
                            <td className={`px-5 py-3.5 ${t.text3}`}>
                              {user.age || "\u2014"}
                            </td>
                            <td
                              className={`px-5 py-3.5 ${t.text3} max-w-[200px] truncate`}
                            >
                              {user.basicInfo || "\u2014"}
                            </td>
                            <td className={`px-5 py-3.5 ${t.text4} text-xs`}>
                              {user.createdAt
                                ? new Date(user.createdAt).toLocaleDateString()
                                : "\u2014"}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ========== ASSESSMENTS ========== */}
          {activeTab === "assessments" && (
            <div className="space-y-5 animate-fadeIn">
              <div>
                <h1
                  className={`text-2xl font-bold bg-gradient-to-r ${t.gradientHeading} bg-clip-text text-transparent`}
                >
                  Assessment Activity
                </h1>
                <p className={`text-sm ${t.text3} mt-1`}>
                  {snapshot.assessments.length} total assessment
                  {snapshot.assessments.length !== 1 ? "s" : ""}
                </p>
              </div>

              <div
                className={`rounded-2xl border ${t.cardBorder} overflow-hidden`}
                style={
                  isDark
                    ? { background: t.cardBg, backdropFilter: "blur(12px)" }
                    : { background: t.cardBg }
                }
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr
                        className={`border-b ${isDark ? "border-white/[0.06]" : "border-gray-200"} ${t.tableHeadBg}`}
                      >
                        {[
                          "Patient",
                          "Score",
                          "Severity",
                          "Recordings",
                          "Date",
                        ].map((h) => (
                          <th
                            key={h}
                            className={`text-left px-5 py-3.5 text-[11px] font-semibold ${t.text3} uppercase tracking-widest`}
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.assessments.length === 0 ? (
                        <tr>
                          <td colSpan={5}>
                            <EmptyState t={t} text="No assessments yet" />
                          </td>
                        </tr>
                      ) : (
                        snapshot.assessments.map((a) => (
                          <tr
                            key={a.id}
                            className={`border-b ${t.tableRowBorder} ${t.tableRowHover} transition-colors`}
                          >
                            <td className="px-5 py-3.5">
                              <p className={`font-medium ${t.text1}`}>
                                {a.userName || "Anonymous"}
                              </p>
                              <p className={`text-[11px] ${t.text4}`}>
                                {a.email}
                              </p>
                            </td>
                            <td className="px-5 py-3.5">
                              <span className={`font-bold ${t.text1}`}>
                                {a.score}
                              </span>
                              <span className={t.text4}>/24</span>
                            </td>
                            <td className="px-5 py-3.5">
                              <SeverityBadge severity={a.severity} t={t} />
                            </td>
                            <td className={`px-5 py-3.5 ${t.text3}`}>
                              {a.recordingCount || 0}
                            </td>
                            <td className={`px-5 py-3.5 ${t.text4} text-xs`}>
                              {a.createdAt
                                ? new Date(a.createdAt).toLocaleString()
                                : "\u2014"}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* ========== MONITORING ========== */}
          {activeTab === "monitoring" && <MonitoringTab t={t} />}

          {/* ========== SETTINGS ========== */}
          {activeTab === "settings" && (
            <div className="space-y-8 max-w-3xl animate-fadeIn">
              <div>
                <h1
                  className={`text-2xl font-bold bg-gradient-to-r ${t.gradientHeading} bg-clip-text text-transparent`}
                >
                  Settings
                </h1>
                <p className={`text-sm ${t.text3} mt-1`}>
                  Manage your admin panel preferences and view system info
                </p>
              </div>

              {/* Appearance */}
              <section className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 ${t.accent}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M4.098 19.902a3.75 3.75 0 005.304 0l6.401-6.402M6.75 21A3.75 3.75 0 013 17.25V4.125C3 3.504 3.504 3 4.125 3h5.25c.621 0 1.125.504 1.125 1.125v4.072M6.75 21a3.75 3.75 0 003.75-3.75V8.197M6.75 21h13.125c.621 0 1.125-.504 1.125-1.125v-5.25c0-.621-.504-1.125-1.125-1.125h-4.072M10.5 8.197l2.88-2.88c.438-.439 1.15-.439 1.59 0l3.712 3.713c.44.44.44 1.152 0 1.59l-2.879 2.88M6.75 17.25h.008v.008H6.75v-.008z"
                    />
                  </svg>
                  <h2 className={`text-base font-semibold ${t.text1}`}>
                    Appearance
                  </h2>
                </div>
                <div
                  className={`rounded-2xl border p-6 ${t.settingSectionBg} transition-shadow ${t.settingSectionHover}`}
                >
                  <p className={`text-sm font-medium ${t.text2} mb-1`}>Theme</p>
                  <p className={`text-xs ${t.text3} mb-5`}>
                    Choose how the admin dashboard looks for you
                  </p>
                  <div className="grid sm:grid-cols-2 gap-4">
                    {/* Light card */}
                    <button
                      onClick={() => setTheme("light")}
                      className={`relative rounded-xl border-2 p-4 text-left transition-all ${!isDark ? t.themeCardActiveBorder : t.themeCardBorder} ${t.themeCardBg}`}
                    >
                      <div className="w-full h-20 rounded-lg mb-3 overflow-hidden bg-[#F4F6F9] border border-gray-200 p-2">
                        <div className="flex h-full gap-1.5">
                          <div className="w-8 bg-white rounded-md border border-gray-100" />
                          <div className="flex-1 space-y-1.5">
                            <div className="h-2 bg-gray-200 rounded w-3/4" />
                            <div className="flex gap-1.5">
                              <div className="h-6 flex-1 bg-white rounded border border-gray-100" />
                              <div className="h-6 flex-1 bg-white rounded border border-gray-100" />
                            </div>
                          </div>
                        </div>
                      </div>
                      <p className={`text-sm font-semibold ${t.text1}`}>
                        Light
                      </p>
                      <p className={`text-xs ${t.text3}`}>
                        Clean and bright interface
                      </p>
                      {!isDark && (
                        <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                          <svg
                            className="w-3 h-3 text-white"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={3}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M4.5 12.75l6 6 9-13.5"
                            />
                          </svg>
                        </div>
                      )}
                    </button>
                    {/* Dark card */}
                    <button
                      onClick={() => setTheme("dark")}
                      className={`relative rounded-xl border-2 p-4 text-left transition-all ${isDark ? t.themeCardActiveBorder : t.themeCardBorder} ${t.themeCardBg}`}
                    >
                      <div className="w-full h-20 rounded-lg mb-3 overflow-hidden bg-[#0B0F19] border border-white/10 p-2">
                        <div className="flex h-full gap-1.5">
                          <div className="w-8 bg-white/5 rounded-md border border-white/10" />
                          <div className="flex-1 space-y-1.5">
                            <div className="h-2 bg-white/10 rounded w-3/4" />
                            <div className="flex gap-1.5">
                              <div className="h-6 flex-1 bg-white/5 rounded border border-white/10" />
                              <div className="h-6 flex-1 bg-white/5 rounded border border-white/10" />
                            </div>
                          </div>
                        </div>
                      </div>
                      <p className={`text-sm font-semibold ${t.text1}`}>Dark</p>
                      <p className={`text-xs ${t.text3}`}>
                        Easier on the eyes at night
                      </p>
                      {isDark && (
                        <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                          <svg
                            className="w-3 h-3 text-white"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                            strokeWidth={3}
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              d="M4.5 12.75l6 6 9-13.5"
                            />
                          </svg>
                        </div>
                      )}
                    </button>
                  </div>
                </div>
              </section>

              {/* Account */}
              <section className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 ${t.accent}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
                    />
                  </svg>
                  <h2 className={`text-base font-semibold ${t.text1}`}>
                    Account
                  </h2>
                </div>
                <div
                  className={`rounded-2xl border p-6 ${t.settingSectionBg} transition-shadow ${t.settingSectionHover}`}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500 to-cyan-600 flex items-center justify-center text-xl font-bold text-white shadow-lg shadow-emerald-500/20">
                      A
                    </div>
                    <div className="flex-1">
                      <p className={`text-base font-semibold ${t.text1}`}>
                        {session.adminId}
                      </p>
                      <p className={`text-sm ${t.text3}`}>Super Admin</p>
                    </div>
                    <button
                      onClick={handleSignOut}
                      className={`px-4 py-2 rounded-xl text-xs font-semibold border transition-all ${t.signOut}`}
                    >
                      Sign Out
                    </button>
                  </div>
                </div>
              </section>

              {/* System Status */}
              <section className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 ${t.accent}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M5.25 14.25h13.5m-13.5 0a3 3 0 01-3-3m3 3a3 3 0 100 6h13.5a3 3 0 100-6m-16.5-3a3 3 0 013-3h13.5a3 3 0 013 3m-19.5 0a4.5 4.5 0 01.9-2.7L5.737 5.1a3.375 3.375 0 012.7-1.35h7.126c1.062 0 2.062.5 2.7 1.35l2.587 3.45a4.5 4.5 0 01.9 2.7m0 0a3 3 0 01-3 3m0 3h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008zm-3 6h.008v.008h-.008v-.008zm0-6h.008v.008h-.008v-.008z"
                    />
                  </svg>
                  <h2 className={`text-base font-semibold ${t.text1}`}>
                    System Status
                  </h2>
                </div>
                <div
                  className={`rounded-2xl border p-6 ${t.settingSectionBg} transition-shadow ${t.settingSectionHover}`}
                >
                  <div className="space-y-3">
                    {[
                      {
                        name: "Backend API",
                        port: "8000",
                        desc: "FastAPI application server",
                      },
                      {
                        name: "ML Model Service",
                        port: "8001",
                        desc: "Depression prediction pipeline",
                      },
                      {
                        name: "Frontend",
                        port: "5173",
                        desc: "React + Vite dev server",
                      },
                      {
                        name: "Database",
                        port: "SQLite",
                        desc: "Async SQLAlchemy storage",
                      },
                    ].map((svc) => (
                      <div
                        key={svc.name}
                        className={`flex items-center justify-between py-3 px-4 rounded-xl border ${t.svcRowBg}`}
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <p className={`text-sm font-medium ${t.text2}`}>
                              {svc.name}
                            </p>
                            <span
                              className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${isDark ? "bg-white/5 text-white/40" : "bg-gray-100 text-gray-400"}`}
                            >
                              {svc.port}
                            </span>
                          </div>
                          <p className={`text-xs ${t.text4} mt-0.5`}>
                            {svc.desc}
                          </p>
                        </div>
                        <span
                          className={`flex items-center gap-1.5 text-[11px] font-semibold ${isDark ? "text-emerald-400" : "text-emerald-600"}`}
                        >
                          <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-50" />
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                          </span>
                          Operational
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              {/* Platform Insights */}
              <section className="space-y-4">
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 ${t.accent}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 013 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z"
                    />
                  </svg>
                  <h2 className={`text-base font-semibold ${t.text1}`}>
                    Platform Insights
                  </h2>
                </div>
                <div
                  className={`rounded-2xl border p-6 ${t.settingSectionBg} transition-shadow ${t.settingSectionHover}`}
                >
                  <div className="grid sm:grid-cols-3 gap-4">
                    {[
                      {
                        label: "Doctor : Patient Ratio",
                        value:
                          snapshot.totals.patients > 0
                            ? `1 : ${Math.round(snapshot.totals.patients / Math.max(snapshot.totals.doctors, 1))}`
                            : "N/A",
                      },
                      {
                        label: "Avg Assessments / Patient",
                        value:
                          snapshot.totals.patients > 0
                            ? (
                                snapshot.totals.assessments /
                                snapshot.totals.patients
                              ).toFixed(1)
                            : "N/A",
                      },
                      {
                        label: "Total Platform Users",
                        value: snapshot.totals.users.toString(),
                      },
                    ].map((s) => (
                      <div
                        key={s.label}
                        className={`p-4 rounded-xl border ${t.svcRowBg} text-center`}
                      >
                        <p
                          className={`text-[11px] ${t.text4} mb-2 uppercase tracking-wider font-medium`}
                        >
                          {s.label}
                        </p>
                        <p
                          className={`text-2xl font-bold bg-gradient-to-r ${t.insightValue} bg-clip-text text-transparent`}
                        >
                          {s.value}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              {/* About */}
              <section className="space-y-4 pb-6">
                <div className="flex items-center gap-2">
                  <svg
                    className={`w-5 h-5 ${t.accent}`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={1.5}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z"
                    />
                  </svg>
                  <h2 className={`text-base font-semibold ${t.text1}`}>
                    About
                  </h2>
                </div>
                <div
                  className={`rounded-2xl border p-6 ${t.settingSectionBg} transition-shadow ${t.settingSectionHover}`}
                >
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/15 flex-shrink-0">
                      <svg
                        className="w-6 h-6 text-white"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={2}
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
                        />
                      </svg>
                    </div>
                    <div>
                      <p className={`text-base font-semibold ${t.text1}`}>
                        MindScope
                      </p>
                      <p className={`text-xs ${t.text3} mt-0.5`}>
                        Depression Screening Platform
                      </p>
                      <p className={`text-xs ${t.text4} mt-2 leading-relaxed`}>
                        PHQ-8 questionnaire with AI-powered voice analysis for
                        comprehensive mental health screening. Built with React,
                        FastAPI, and PyTorch.
                      </p>
                      <div className="flex items-center gap-3 mt-3">
                        <span
                          className={`text-[10px] font-mono px-2 py-1 rounded-lg ${isDark ? "bg-white/5 text-white/40 border border-white/10" : "bg-gray-100 text-gray-400 border border-gray-200"}`}
                        >
                          v1.0.0
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </main>
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fadeIn { animation: fadeIn 0.3s ease-out; }
      `}</style>
    </div>
  );
}

/* ---- Shared components ---- */

function GlassCard({
  t,
  isDark,
  title,
  subtitle,
  action,
  actionLabel,
  children,
}) {
  return (
    <div
      className={`rounded-2xl border ${t.cardBorder} p-5`}
      style={
        isDark
          ? { background: t.cardBg, backdropFilter: "blur(12px)" }
          : { background: t.cardBg }
      }
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className={`text-sm font-semibold ${t.text1}`}>{title}</h2>
          {subtitle && (
            <p className={`text-[11px] ${t.text4} mt-0.5`}>{subtitle}</p>
          )}
        </div>
        {action && (
          <button
            onClick={action}
            className={`text-[11px] ${t.accent} font-medium transition-colors opacity-80 hover:opacity-100`}
          >
            {actionLabel || "View all"} &rarr;
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

function Avatar({ name, role }) {
  const bg =
    role === "doctor"
      ? "bg-gradient-to-br from-emerald-500 to-emerald-700"
      : "bg-gradient-to-br from-violet-500 to-violet-700";
  return (
    <div
      className={`w-8 h-8 rounded-lg ${bg} flex items-center justify-center text-[11px] font-bold text-white shadow-sm`}
    >
      {name?.[0]?.toUpperCase() || "U"}
    </div>
  );
}

function RoleBadge({ role, t }) {
  const style = role === "doctor" ? t.badgeDoctor : t.badgePatient;
  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-lg text-[11px] font-semibold border capitalize ${style}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${role === "doctor" ? "bg-emerald-500" : "bg-violet-500"}`}
      />
      {role}
    </span>
  );
}

function SeverityBadge({ severity, score, t }) {
  const s = severity?.toLowerCase() || "";
  const style = s.includes("severe")
    ? t.sevSevere
    : s.includes("moderate")
      ? t.sevModerate
      : s.includes("mild")
        ? t.sevMild
        : t.sevNone;
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-lg text-[11px] font-semibold border ${style}`}
    >
      {severity || "None"}
      {score != null ? ` \u00b7 ${score}/24` : ""}
    </span>
  );
}

function EmptyState({ t, text }) {
  return (
    <div className="flex flex-col items-center justify-center py-10">
      <svg
        className={`w-10 h-10 mb-2 ${t.text4} opacity-40`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={1}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5M10 11.25h4M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z"
        />
      </svg>
      <p className={`text-sm ${t.text3}`}>{text}</p>
    </div>
  );
}
