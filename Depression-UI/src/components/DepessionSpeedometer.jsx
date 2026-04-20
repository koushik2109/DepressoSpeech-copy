import { useEffect, useState } from 'react';

export default function DepessionSpeedometer({ score = 0, level = 'Minimal', maxScore = 24 }) {
  const severityRanges = [
    { label: 'Minimal', min: 0, max: 4, color: '#52B788' },
    { label: 'Mild', min: 5, max: 9, color: '#95D5B2' },
    { label: 'Moderate', min: 10, max: 14, color: '#FBBF24' },
    { label: 'Moderately Severe', min: 15, max: 19, color: '#FB923C' },
    { label: 'Severe', min: 20, max: 24, color: '#EF4444' },
  ];

  const normalizedScore = Math.max(0, Math.min(maxScore, score));
  const targetAngle = (normalizedScore / maxScore) * 180 - 180;
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

  // Needle tip position for the score label
  const needleRad = (needleAngle) * (Math.PI / 180);
  const cx = 150;
  const cy = 145;
  const needleLength = 72;

  return (
    <div className="flex flex-col items-center justify-center w-full">
      <div className="relative w-full max-w-[22rem]">
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
            const tickAngle = (value / maxScore) * 180 - 180;
            const tickRad = tickAngle * (Math.PI / 180);
            const x1 = cx + 115 * Math.cos(tickRad);
            const y1 = cy + 115 * Math.sin(tickRad);
            const x2 = cx + 125 * Math.cos(tickRad);
            const y2 = cy + 125 * Math.sin(tickRad);
            const lx = cx + 135 * Math.cos(tickRad);
            const ly = cy + 135 * Math.sin(tickRad);
            return (
              <g key={value}>
                <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#D1D1D1" strokeWidth="2" />
                <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" style={{ fontSize: '10px', fontWeight: 500 }} fill="#B5B5B5">
                  {value}
                </text>
              </g>
            );
          })}

          {/* Needle */}
          <g transform={`rotate(${needleAngle} ${cx} ${cy})`} filter="url(#needleShadow)">
            <line x1={cx} y1={cy} x2={cx - needleLength} y2={cy} stroke={currentColor} strokeWidth="4" strokeLinecap="round" />
          </g>

          {/* Center hub — small dot */}
          <circle cx={cx} cy={cy} r="7" fill="white" stroke={currentColor} strokeWidth="2.5" />
          <circle cx={cx} cy={cy} r="3" fill={currentColor} />

          {/* Score display — well below the hub */}
          <text x={cx} y="192" textAnchor="middle" style={{ fontSize: '32px', fontWeight: 800, fontFamily: 'Inter, sans-serif', letterSpacing: '-0.02em' }} fill={currentColor}>
            {normalizedScore}
          </text>
          <text x={cx + 20} y="192" textAnchor="start" style={{ fontSize: '13px', fontWeight: 500 }} fill="#B5B5B5">
            /{maxScore}
          </text>
        </svg>
      </div>

      <div className="mt-2 text-center">
        <p className="text-[10px] uppercase tracking-[0.2em] text-[#B5B5B5] font-semibold mb-1">Severity Level</p>
        <p className="text-2xl font-bold" style={{ color: currentColor }}>{level}</p>
      </div>

      <div className="mt-8 w-full">
        <div className="grid grid-cols-5 gap-2">
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
                <p className="text-[11px] font-bold text-[#555]">{range.min}–{range.max}</p>
                <p className="text-[10px] text-[#B5B5B5] leading-tight mt-0.5">{range.label}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
