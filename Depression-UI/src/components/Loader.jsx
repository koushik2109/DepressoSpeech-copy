export default function Loader({ size = 'md', text = '' }) {
  const sizes = {
    sm: 'w-12 h-12',
    md: 'w-16 h-16',
    lg: 'w-20 h-20',
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <div className={`relative ${sizes[size]} flex items-center justify-center`}>
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[#D8F3DC] via-[#B7E4C7] to-[#F7F7F2] animate-pulse" />
        <div className="absolute inset-2 rounded-full border-4 border-white/90 border-t-[#2D6A4F] border-r-[#52B788] animate-spin shadow-sm" />
        <div className="relative w-2.5 h-2.5 rounded-full bg-[#2D6A4F] shadow-[0_0_18px_rgba(45,106,79,0.5)]" />
      </div>
      {text && (
        <p className="text-sm text-[#777] animate-pulse text-center max-w-xs">{text}</p>
      )}
    </div>
  );
}
