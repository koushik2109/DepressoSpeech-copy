import { useState, useEffect, useCallback } from "react";
import {
  AreaChart, Area,
  BarChart, Bar,
  LineChart, Line,
  PieChart, Pie, Cell,
  CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { getAdminMetrics, getMLHealth } from "../services/api.js";

const PIE_COLORS = ["#52B788", "#95D5B2", "#FBBF24", "#FB923C", "#EF4444"];

export default function MonitoringTab({ t }) {
  const [metrics, setMetrics] = useState(null);
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [m, h] = await Promise.all([getAdminMetrics(), getMLHealth()]);
      setMetrics(m);
      setHealth(h);
    } catch (e) {
      console.error("Monitoring fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 60000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className={`w-8 h-8 border-3 border-t-transparent rounded-full animate-spin ${t.accent}`} style={{ borderTopColor: "transparent" }} />
      </div>
    );
  }

  const timeline = metrics?.timeline || [];
  const severityDist = metrics?.severityDistribution || [];
  const recentPredictions = metrics?.recentPredictions || [];

  // Reusable card wrapper
  const Card = ({ children, className = "" }) => (
    <div className={`rounded-2xl border backdrop-blur-sm p-5 ${t.cardBorder} ${className}`}
      style={{ backgroundColor: t.cardBg }}>
      {children}
    </div>
  );

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Health Status Cards */}
      <div className="grid sm:grid-cols-2 gap-4">
        {[
          { label: "Backend API", data: health?.backend },
          { label: "ML Model", data: health?.mlModel },
        ].map((svc) => {
          const isHealthy = svc.data?.status === "healthy";
          return (
            <Card key={svc.label}>
              <div className="flex items-center justify-between mb-3">
                <h3 className={`text-sm font-semibold ${t.text1}`}>{svc.label}</h3>
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${isHealthy ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`} />
                  <span className={`text-xs font-medium ${isHealthy ? "text-emerald-500" : "text-red-500"}`}>
                    {svc.data?.status || "unknown"}
                  </span>
                </div>
              </div>
              <div className={`text-xs space-y-1 ${t.text3}`}>
                {svc.data?.latency_ms != null && <p>Latency: {svc.data.latency_ms}ms</p>}
                {svc.data?.device && <p>Device: {svc.data.device}</p>}
                {svc.data?.modelLoaded != null && <p>Model: {svc.data.modelLoaded ? "Loaded" : "Not loaded"}</p>}
                {svc.data?.error && <p className="text-red-400 break-all">{svc.data.error}</p>}
              </div>
            </Card>
          );
        })}
      </div>

      {/* Request Rate + Latency + Error Rate */}
      <div className="grid lg:grid-cols-3 gap-4">
        <Card>
          <h3 className={`text-sm font-semibold mb-4 ${t.text1}`}>Request Rate (24h)</h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timeline} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={false} />
                <YAxis tick={{ fill: "#9CA3AF", fontSize: 10 }} tickLine={false} axisLine={false} />
                <Tooltip />
                <Area type="monotone" dataKey="requests" stroke="#52B788" fill="#52B788" fillOpacity={0.15} strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <h3 className={`text-sm font-semibold mb-4 ${t.text1}`}>Avg Latency (ms)</h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeline} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={false} />
                <YAxis tick={{ fill: "#9CA3AF", fontSize: 10 }} tickLine={false} axisLine={false} />
                <Tooltip />
                <Line type="monotone" dataKey="avgLatency" stroke="#7C3AED" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <h3 className={`text-sm font-semibold mb-4 ${t.text1}`}>Error Rate (%)</h3>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={timeline} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={false} />
                <YAxis tick={{ fill: "#9CA3AF", fontSize: 10 }} tickLine={false} axisLine={false} />
                <Tooltip />
                <Bar dataKey="errorRate" fill="#EF4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      {/* ML Severity Distribution + Recent Predictions */}
      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <h3 className={`text-sm font-semibold mb-4 ${t.text1}`}>ML Prediction Distribution</h3>
          {severityDist.length > 0 ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={severityDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={75}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={10}>
                    {severityDist.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className={`text-sm text-center py-12 ${t.text4}`}>No ML predictions yet</p>
          )}
        </Card>

        <Card>
          <h3 className={`text-sm font-semibold mb-4 ${t.text1}`}>Recent ML Predictions</h3>
          {recentPredictions.length > 0 ? (
            <div className="overflow-auto max-h-52">
              <table className="w-full text-xs">
                <thead>
                  <tr className={t.text3}>
                    <th className="text-left pb-2 font-medium">Score</th>
                    <th className="text-left pb-2 font-medium">Severity</th>
                    <th className="text-left pb-2 font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {recentPredictions.map((p) => (
                    <tr key={p.id} className={`border-t ${t.tableRowBorder}`}>
                      <td className={`py-1.5 font-semibold ${t.text1}`}>{p.mlScore?.toFixed(1)}</td>
                      <td className={`py-1.5 ${t.text3}`}>{p.mlSeverity}</td>
                      <td className={`py-1.5 ${t.text3}`}>
                        {p.createdAt ? new Date(p.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short" }) : "\u2014"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className={`text-sm text-center py-12 ${t.text4}`}>No predictions yet</p>
          )}
        </Card>
      </div>

      <p className={`text-xs text-center ${t.text4}`}>
        Auto-refreshes every 60 seconds
      </p>
    </div>
  );
}
