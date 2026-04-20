/**
 * Card.jsx
 * Reusable card container with soft shadow and rounded corners.
 * Supports optional hover effect and click handler.
 */
export default function Card({
  children,
  className = "",
  hover = false,
  onClick = null,
  selected = false,
}) {
  return (
    <div
      onClick={onClick}
      className={`
        bg-white/80 backdrop-blur-sm rounded-2xl border border-[#E8E8E8]/60 p-6
        shadow-[0_2px_20px_rgba(0,0,0,0.06)]
        ${hover ? "hover:shadow-[0_12px_40px_rgba(0,0,0,0.1)] hover:-translate-y-1 cursor-pointer" : ""}
        ${selected ? "ring-2 ring-[#2D6A4F] border-[#2D6A4F] shadow-[0_12px_40px_rgba(0,0,0,0.1)]" : ""}
        ${onClick ? "cursor-pointer" : ""}
        transition-all duration-300
        ${className}
      `}
    >
      {children}
    </div>
  );
}
