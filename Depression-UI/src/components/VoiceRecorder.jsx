import { useEffect, useMemo, useRef, useState } from 'react';

const MAX_RECORDING_SECONDS = 120;
const MIN_RECORDING_SECONDS = 5;
const FFT_SIZE = 2048;
const BAR_COUNT = 64;

export default function VoiceRecorder({ onRecordingComplete, onRecordingCleared }) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [audioURL, setAudioURL] = useState('');
  const [recordingBlobSize, setRecordingBlobSize] = useState(0);
  const [level, setLevel] = useState(0);
  const [hasRecording, setHasRecording] = useState(false);

  const timerRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);
  const dataArrayRef = useRef(null);
  const freqDataArrayRef = useRef(null);
  const recordedBlobRef = useRef(null);
  const audioURLRef = useRef('');
  const secondsRef = useRef(0);

  const formatTime = (value) => {
    const minutes = Math.floor(value / 60).toString().padStart(2, '0');
    const secondsValue = (value % 60).toString().padStart(2, '0');
    return `${minutes}:${secondsValue}`;
  };

  const clearCurrentRecording = () => {
    if (audioURLRef.current) {
      URL.revokeObjectURL(audioURLRef.current);
      audioURLRef.current = '';
    }
    setAudioURL('');
    setHasRecording(false);
    setRecordingBlobSize(0);
    recordedBlobRef.current = null;
    if (onRecordingCleared) {
      onRecordingCleared();
    }
  };

  const drawIdleLine = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext('2d');
    if (!context) return;

    const width = canvas.width;
    const height = canvas.height;

    context.clearRect(0, 0, width, height);
    context.fillStyle = '#FAFAF7';
    context.fillRect(0, 0, width, height);

    context.strokeStyle = '#D8F3DC';
    context.lineWidth = Math.max(2, width / 240);
    context.lineCap = 'round';
    context.beginPath();
    context.moveTo(0, height / 2);
    context.lineTo(width, height / 2);
    context.stroke();
  };

  const drawWaveform = () => {
    const canvas = canvasRef.current;
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;
    const freqDataArray = freqDataArrayRef.current;
    if (!canvas || !analyser || !dataArray || !freqDataArray) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    analyser.getByteTimeDomainData(dataArray);

    const width = canvas.width;
    const height = canvas.height;

    context.clearRect(0, 0, width, height);
    const background = context.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, '#FAFAF7');
    background.addColorStop(1, '#F0FAF4');
    context.fillStyle = background;
    context.fillRect(0, 0, width, height);

    analyser.getByteFrequencyData(freqDataArray);
    const barWidth = width / BAR_COUNT;
    for (let i = 0; i < BAR_COUNT; i += 1) {
      const dataIndex = Math.floor((i / BAR_COUNT) * freqDataArray.length);
      const magnitude = freqDataArray[dataIndex] / 255;
      const barHeight = Math.max(2, magnitude * (height * 0.8));
      const x = i * barWidth;
      const y = (height - barHeight) / 2;

      const barGradient = context.createLinearGradient(0, y, 0, y + barHeight);
      barGradient.addColorStop(0, '#52B788');
      barGradient.addColorStop(1, '#2D6A4F');
      context.fillStyle = barGradient;
      context.fillRect(x + barWidth * 0.15, y, barWidth * 0.7, barHeight);
    }

    const centerY = height / 2;
    const sliceWidth = width / dataArray.length;

    context.beginPath();
    context.moveTo(0, centerY);

    for (let i = 0; i < dataArray.length; i += 1) {
      const value = dataArray[i] / 128.0;
      const y = value * centerY;
      const x = i * sliceWidth;
      context.lineTo(x, y);
    }

    context.strokeStyle = '#2D6A4F';
    context.lineWidth = Math.max(2, width / 220);
    context.lineJoin = 'round';
    context.lineCap = 'round';
    context.stroke();

    const sum = freqDataArray.reduce((total, current) => total + current, 0);
    const nextLevel = Math.min(1, sum / (freqDataArray.length * 255));
    setLevel(nextLevel);

    animFrameRef.current = requestAnimationFrame(drawWaveform);
  };

  const stopRecording = () => {
    clearInterval(timerRef.current);
    timerRef.current = null;
    setIsRecording(false);

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }

    drawIdleLine();
  };

  const startRecording = async () => {
    try {
      clearCurrentRecording();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });

      streamRef.current = stream;

      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = FFT_SIZE;
      analyser.smoothingTimeConstant = 0.85;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      audioContextRef.current = audioContext;
      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.fftSize);
      freqDataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);

      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
        .find((candidate) => MediaRecorder.isTypeSupported(candidate)) || 'audio/webm';

      const recorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000,
      });
      mediaRecorderRef.current = recorder;

      const chunks = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: mimeType });
        recordedBlobRef.current = blob;
        setRecordingBlobSize(blob.size);
        setHasRecording(true);

        const previewUrl = URL.createObjectURL(blob);
        audioURLRef.current = previewUrl;
        setAudioURL(previewUrl);

        if (onRecordingComplete) {
          onRecordingComplete(blob, previewUrl, secondsRef.current);
        }
      };

      recorder.start();
      setIsRecording(true);
      setSeconds(0);
      secondsRef.current = 0;
      drawWaveform();

      timerRef.current = setInterval(() => {
        secondsRef.current += 1;
        setSeconds(secondsRef.current);
        if (secondsRef.current >= MAX_RECORDING_SECONDS) {
          stopRecording();
        }
      }, 1000);
    } catch (error) {
      const message = error.name === 'NotAllowedError'
        ? 'Microphone permission was denied. Please allow access and try again.'
        : error.name === 'NotFoundError'
          ? 'No microphone was found on this device.'
          : error.name === 'NotReadableError'
            ? 'The microphone is in use by another application.'
            : 'Microphone access failed. Please try again.';

      alert(message);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;

    const resizeCanvas = () => {
      const parent = canvas.parentElement;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(parent.clientWidth * ratio));
      canvas.height = Math.max(1, Math.floor(parent.clientHeight * ratio));
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      const context = canvas.getContext('2d');
      if (context) {
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
      }
      if (!isRecording) {
        drawIdleLine();
      }
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [isRecording]);

  useEffect(() => () => {
    clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
    if (audioContextRef.current) audioContextRef.current.close();
    if (audioURLRef.current) URL.revokeObjectURL(audioURLRef.current);
  }, []);

  const canPlayback = hasRecording && audioURL;

  return (
    <div className="flex flex-col items-center gap-6 w-full">
      <div className="w-full">
        <p className="text-xs font-medium text-[#777] uppercase tracking-wider mb-2.5">Voice Waveform</p>
        <div className="w-full h-28 rounded-2xl bg-[#FAFAF7] border border-[#E8E8E8] overflow-hidden shadow-sm">
          <canvas ref={canvasRef} className="w-full h-full" />
        </div>
      </div>

      <div className="flex items-center gap-4 w-full">
        {/* Left spacer mirrors the Level block so the timer is truly centred */}
        <div className="w-24 flex items-center justify-start">
          {isRecording && <span className="inline-block w-3 h-3 rounded-full bg-red-500 animate-pulse shadow-lg shadow-red-200" />}
        </div>
        <div className="text-center flex-1">
          <span className="text-3xl font-bold text-[#1B1B1B] font-mono tracking-tight">{formatTime(seconds)}</span>
          <p className="text-xs text-[#B5B5B5] mt-1">{isRecording ? 'Recording' : 'Ready'} • Max {formatTime(MAX_RECORDING_SECONDS)}</p>
        </div>
        <div className="w-24 text-right">
          <p className="text-xs uppercase tracking-[0.16em] text-[#777]">Level</p>
          <div className="mt-2 h-2 rounded-full bg-gray-100 overflow-hidden">
            <div className="h-full rounded-full bg-gradient-to-r from-[#52B788] to-[#2D6A4F] transition-all duration-150" style={{ width: `${Math.max(8, level * 100)}%` }} />
          </div>
        </div>
      </div>

      <div className="relative">
        {isRecording && <div className="absolute -inset-3 rounded-full bg-red-400/20 animate-pulse" />}
        <button
          type="button"
          onClick={isRecording ? stopRecording : startRecording}
          className={`relative w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 shadow-lg hover:shadow-xl active:scale-95 font-semibold text-white text-sm ${isRecording ? 'bg-red-500 hover:bg-red-600 shadow-red-200' : 'bg-[#2D6A4F] hover:bg-[#1B3A2D] shadow-[#2D6A4F]/20'}`}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isRecording ? (
            <div className="w-7 h-7 rounded-md bg-white" />
          ) : (
            <svg className="w-9 h-9 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6.75 6.75 0 006.75-6.75V8.25a6.75 6.75 0 10-13.5 0V12A6.75 6.75 0 0012 18.75zm0 0v2.5m-3.75 0h7.5" />
            </svg>
          )}
        </button>
      </div>

      <div className="text-center w-full">
        {isRecording ? (
          <p className="text-sm font-medium text-[#777]">
            Recording in progress. Tap stop when you are finished.
          </p>
        ) : canPlayback && seconds >= MIN_RECORDING_SECONDS ? (
          <div className="space-y-3">
            <p className="text-sm font-medium text-[#777]">Recording saved ({seconds}s) — you can playback or re-record.</p>
            <div className="bg-white p-4 rounded-2xl border border-[#E8E8E8] shadow-sm">
              <audio controls controlsList="nodownload" src={audioURL} className="w-full h-12" />
            </div>
            <p className="text-xs text-[#B5B5B5]">File size: {(recordingBlobSize / 1024).toFixed(1)} KB</p>
          </div>
        ) : canPlayback && seconds < MIN_RECORDING_SECONDS ? (
          <p className="text-sm text-[#777]">
            Recording too short ({seconds}s). Minimum {MIN_RECORDING_SECONDS}s required. Please re-record.
          </p>
        ) : (
          <p className="text-sm font-medium text-[#777]">
            Tap the microphone to begin recording. Speak naturally and clearly.
          </p>
        )}
      </div>

      {hasRecording && audioURL && (
        <button
          type="button"
          onClick={startRecording}
          className="text-sm text-[#2D6A4F] hover:text-[#1B3A2D] font-semibold flex items-center gap-2 transition-colors group hover:bg-[#D8F3DC] px-4 py-2 rounded-lg"
        >
          <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182" />
          </svg>
          Re-record voice
        </button>
      )}
    </div>
  );
}
