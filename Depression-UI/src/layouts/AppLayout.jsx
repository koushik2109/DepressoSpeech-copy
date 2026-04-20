import { Outlet } from "react-router-dom";
import Navbar from "../components/Navbar";

function AppLayout() {
  return (
    <div className="min-h-screen bg-[#F8FAF9] text-slate-800">
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_15%_20%,rgba(31,122,102,0.08),transparent_32%),radial-gradient(circle_at_80%_10%,rgba(83,169,151,0.07),transparent_34%),linear-gradient(to_bottom,#f8faf9,#f8faf9)]" />
      </div>
      <Navbar />
      <main className="pt-20 md:pt-24">
        <Outlet />
      </main>
    </div>
  );
}

export default AppLayout;
