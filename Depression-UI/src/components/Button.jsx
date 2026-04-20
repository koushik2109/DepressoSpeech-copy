/**
 * Button.jsx
 * Reusable button with primary, secondary, and outline variants.
 */
export default function Button({
  children,
  onClick,
  variant = "primary",
  size = "md",
  disabled = false,
  fullWidth = false,
  type = "button",
  className = "",
}) {
  const base =
    "inline-flex items-center justify-center font-semibold rounded-xl transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#2D6A4F]/50";

  const variants = {
    primary:
      "bg-gradient-to-r from-[#1B3A2D] to-[#2D6A4F] text-white hover:from-[#2D6A4F] hover:to-[#1B3A2D] shadow-md hover:shadow-lg hover:shadow-[#2D6A4F]/20 active:scale-[0.98]",
    secondary:
      "bg-gradient-to-r from-[#D8F3DC] to-[#B7E4C7] text-[#1B3A2D] hover:from-[#B7E4C7] hover:to-[#D8F3DC] shadow-sm hover:shadow-md active:scale-[0.98]",
    outline:
      "border-2 border-[#2D6A4F]/30 text-[#2D6A4F] hover:bg-[#D8F3DC]/50 hover:border-[#2D6A4F] active:scale-[0.98]",
    ghost: "text-[#777] hover:text-[#1B1B1B] hover:bg-gray-50",
  };

  const sizes = {
    sm: "px-5 py-2.5 text-sm gap-1.5",
    md: "px-7 py-3.5 text-base gap-2",
    lg: "px-9 py-4 text-base gap-2.5",
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`
        ${base}
        ${variants[variant]}
        ${sizes[size]}
        ${fullWidth ? "w-full" : ""}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
        ${className}
      `}
    >
      {children}
    </button>
  );
}
