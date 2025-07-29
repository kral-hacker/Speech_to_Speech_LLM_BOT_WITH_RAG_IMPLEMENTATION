import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import tempfile
import os

# === Configuration ===
SAMPLE_RATE = 16000
DURATION = 2.0  # recording duration in seconds
SILENCE_PADDING = 0.5  # extra silence after speech
DEVICE = "cuda" if os.environ.get("USE_GPU") else "cpu"

# === Load Whisper model ===
model = WhisperModel("base.en", device=DEVICE, compute_type="int8")

print("🎙️ Speak now. Press Ctrl+C to stop.")

try:
    while True:
        # Record audio
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        # Add silence padding to ensure end-of-sentence is captured
        silence = np.zeros((int(SILENCE_PADDING * SAMPLE_RATE), 1), dtype='float32')
        padded_audio = np.concatenate((audio, silence), axis=0)

        # Write to a temp WAV file (Windows-safe)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, padded_audio, SAMPLE_RATE)
            path = tmp.name

        try:
            # Transcribe with voice activity detection
            segments, _ = model.transcribe(path, beam_size=1, vad_filter=True)
            for seg in segments:
                print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
        except Exception as e:
            print("❌ Transcription error:", e)
        finally:
            if os.path.exists(path):
                os.remove(path)

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
