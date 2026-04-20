/**
 * ChartPanel.jsx
 * Displays Recharts bar chart for response pattern in the results page.
 */
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const defaultData = [
  { name: 'Sadness', value: 72, color: '#2D6A4F' },
  { name: 'Anxiety', value: 58, color: '#40916C' },
  { name: 'Fatigue', value: 65, color: '#52B788' },
  { name: 'Hopelessness', value: 45, color: '#74C69D' },
  { name: 'Irritability', value: 38, color: '#95D5B2' },
  { name: 'Concentration', value: 52, color: '#B7E4C7' },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white px-4 py-3 rounded-lg shadow-elevated border border-[#E8E8E8]">
        <p className="text-sm font-medium text-[#1B1B1B]">{label}</p>
        <p className="text-sm text-[#2D6A4F] font-semibold">{payload[0].value}%</p>
      </div>
    );
  }
  return null;
};

export default function ChartPanel({ data = defaultData, title = 'Emotional Indicators' }) {
  return (
    <div className="bg-white rounded-xl border border-[#E8E8E8] p-6 shadow-card">
      {title && <h3 className="text-base font-semibold text-[#1B1B1B] mb-6">{title}</h3>}
      <div className="w-full h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#F0FAF4" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: '#777', fontSize: 12 }}
              axisLine={{ stroke: '#E8E8E8' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#777', fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: '#FAFAF7' }} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={36}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color || '#2D6A4F'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
