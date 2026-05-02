import { useEffect, useState } from 'react';

export default function DepessionSpeedometer({ score = 0, level = 'Minimal', maxScore = 24 }) {
  const severityRanges = [
    { label: 'Minimal', min: 0, max: 4, color: '#52B788' },
    { label: 'Mild', min: 5, max: 9, color: '#95D5B2' },
    { label: 'Moderate', min: 10, max: 14, color: '#FBBF24' },
    { label: 'Moderately Severe', min: 15, max: 19, color: '#FB923C' },
    { label: 'Severe', min: 20, max: 24, color: '#EF4444' },
  ];

  const safeMaxScore = Math.max(Number(maxScore) || 24, 1);
  const normalizedScore = Math.max(0, Math.min(safeMaxScore, Number(score) || 0));
  const targetAngle = (normalizedScore / safeMaxScore) * 180 - 180;
  const currentRange = severityRanges.find((range) => normalizedScore >= range.min && normalizedScore <= range.max) || severityRanges[0];
  const currentColor = currentRange.color;
  const [needleAngle, setNeedleAngle] = useState(-180);

  useEffect(() => {
    let frame;
    let start;
    const startAngle = -180;
    const duration = 1100;

    const animate = (timestamp) => {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setNeedleAngle(startAngle + (targetAngle - startAngle) * eased);
      if (progress < 1) {
        frame = requestAnimationFrame(animate);
      }
    };

    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [targetAngle]);

  const cx = 150;
  const cy = 145;
  const needleLength = 86;

  return (
    <div className="flex flex-col items-center justify-center w-full">
      <div className="relative w-full max-w-[26rem]">
        <svg viewBox="0 0 300 210" className="w-full" style={{ filter: 'drop-shadow(0 8px 20px rgba(15, 23, 42, 0.05))' }}>
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#52B788" />
              <stop offset="25%" stopColor="#95D5B2" />
              <stop offset="50%" stopColor="#FBBF24" />
              <stop offset="75%" stopColor="#FB923C" />
              <stop offset="100%" stopColor="#EF4444" />
            </linearGradient>
            <filter id="needleShadow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="1" stdDeviation="2" floodColor={currentColor} floodOpacity="0.3" />
            </filter>
          </defs>

          {/* Background track */}
          <path d="M 35 145 A 115 115 0 0 1 265 145" fill="none" stroke="#F0F0F0" strokeWidth="18" strokeLinecap="round" />

          {/* Coloured gauge arc */}
          <path d="M 35 145 A 115 115 0 0 1 265 145" fill="none" stroke="url(#gaugeGradient)" strokeWidth="18" strokeLinecap="round" />

          {/* Tick marks with labels */}
          {[0, 4, 9, 14, 19, 24].map((value) => {
            const tickAngle = (value / safeMaxScore) * 180 - 180;
            const tickRad = tickAngle * (Math.PI / 180);
            const x1 = cx + 115 * Math.cos(tickRad);
            const y1 = cy + 115 * Math.sin(tickRad);
            const x2 = cx + 125 * Math.cos(tickRad);
            const y2 = cy + 125 * Math.sin(tickRad);
            const lx = cx + 135 * Math.cos(tickRad);
            const ly = cy + 135 * Math.sin(tickRad);
            return (
              <g key={value}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#9AA49F" strokeWidth="2.5" />
                <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" style={{ fontSize: '13px', fontWeight: 800 }} fill="#6A766F">
                  {value}
                </text>
              </g>
            );
          })}

          {/* Needle */}
          <g transform={`rotate(${needleAngle} ${cx} ${cy})`} filter="url(#needleShadow)">
            <line x1={cx} y1={cy} x2={cx + needleLength} y2={cy} stroke={currentColor} strokeWidth="5" strokeLinecap="round" />
            <polygon
              points={`${cx + needleLength + 10},${cy} ${cx + needleLength - 3},${cy - 7} ${cx + needleLength - 3},${cy + 7}`}
              fill={currentColor}
            />
          </g>

          {/* Center hub */}
          <circle cx={cx} cy={cy} r="11" fill="white" stroke={currentColor} strokeWidth="3" />
          <circle cx={cx} cy={cy} r="5" fill={currentColor} />

          {/* Score display */}
          <text x={cx} y="196" textAnchor="middle" style={{ fontSize: '31px', fontWeight: 900, fontFamily: 'Inter, sans-serif', letterSpacing: 0 }} fill={currentColor}>
            {Math.round(normalizedScore)}/{safeMaxScore}
          </text>
        </svg>
      </div>

      <div className="mt-3 text-center">
        <p className="text-xs sm:text-sm uppercase tracking-[0.14em] text-[#6A766F] font-bold mb-1">Severity Level</p>
        <p className="text-2xl sm:text-3xl font-black leading-tight" style={{ color: currentColor }}>{level}</p>
      </div>

      <div className="mt-8 w-full">
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
          {severityRanges.map((range) => {
            const active = normalizedScore >= range.min && normalizedScore <= range.max;
            return (
              <div
                key={range.label}
                className="py-3 px-2 rounded-xl border-2 text-center transition-all"
                style={{
                  borderColor: active ? currentColor : '#F0F0F0',
                  backgroundColor: active ? `${currentColor}12` : 'white',
                  boxShadow: active ? `0 2px 8px ${currentColor}20` : 'none',
                }}
              >
                <div className="w-2.5 h-2.5 rounded-full mx-auto mb-1.5" style={{ backgroundColor: range.color }} />
                <p className="text-xs sm:text-sm font-black text-[#34423B]">{range.min}-{range.max}</p>
                <p className="text-[11px] sm:text-xs font-semibold text-[#6A766F] leading-tight mt-1">{range.label}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
