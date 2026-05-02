import { useEffect, useRef, useState, useCallback } from "react";

export function useAudioRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const captureStatsRef = useRef({
    sampleRate: 0,
    blobSize: 0,
  });

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      // Stop any active stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      // Stop recorder if still active
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== "inactive"
      ) {
        try {
          mediaRecorderRef.current.stop();
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  const startTimer = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    setElapsed(0);
    timerRef.current = setInterval(() => {
      setElapsed((s) => s + 1);
    }, 1000);
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);

      // Revoke previous audio URL if exists
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
        setAudioUrl(null);
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mediaRecorder = new MediaRecorder(stream);
      console.debug(
        "[AudioCapture] sampleRate",
        stream.getAudioTracks()[0]?.getSettings?.().sampleRate || 0,
      );
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        stopTimer();
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        captureStatsRef.current = {
          sampleRate:
            stream.getAudioTracks()[0]?.getSettings?.().sampleRate || 0,
          blobSize: blob.size,
        };
        if (!blob.size) {
          setError("Captured audio is empty.");
          return;
        }
        const url = URL.createObjectURL(blob);
        setAudioUrl((prev) => {
          if (prev) URL.revokeObjectURL(prev);
          return url;
        });
        // Stop all tracks to release the mic
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        console.debug(
          "[AudioCapture] recording complete",
          captureStatsRef.current,
        );
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      startTimer();
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      setError("Microphone access denied or not available.");
    }
  }, [audioUrl, startTimer, stopTimer]);

  const stopRecording = useCallback(() => {
    if (
      !mediaRecorderRef.current ||
      mediaRecorderRef.current.state === "inactive"
    )
      return;
    try {
      mediaRecorderRef.current.stop();
    } catch {
      /* ignore */
    }
    setIsRecording(false);
  }, []);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const reset = useCallback(() => {
    stopTimer();
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        /* ignore */
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setIsRecording(false);
    setElapsed(0);
    setError(null);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    chunksRef.current = [];
    mediaRecorderRef.current = null;
  }, [audioUrl, stopTimer]);

  return {
    isRecording,
    elapsed,
    audioUrl,
    error,
    startRecording,
    stopRecording,
    toggleRecording,
    reset,
  };
}
