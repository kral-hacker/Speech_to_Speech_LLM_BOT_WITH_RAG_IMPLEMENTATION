import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import tempfile
import openai
import os

# === CONFIG ===
SAMPLE_RATE = 16000
DURATION = 4.0
SILENCE_PADDING = 1.0
DEVICE = "cuda" if os.environ.get("USE_GPU") else "cpu"

# === Setup Groq API (OpenAI-compatible)
openai.api_key = "##"
openai.api_base = "https://api.groq.com/openai/v1"

# === Load Whisper Model ===
model = WhisperModel("base.en", device=DEVICE, compute_type="int8")

# === Conversation History ===
chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

def get_response_from_groq(prompt, history):
    history.append({"role": "user", "content": prompt})
    try:
        response = openai.ChatCompletion.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # or mistral-7b if preferred
            messages=history,
            temperature=0.7,
            max_tokens=512
        )
        reply = response.choices[0].message["content"]
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"[Error contacting Groq API: {e}]"

# === Start Listening and Chatting ===
print("🎙️ Speak to the bot. Press Ctrl+C to exit.")

try:
    while True:
        # Record your voice
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        # Add silence padding to avoid cutting off words
        silence = np.zeros((int(SILENCE_PADDING * SAMPLE_RATE), 1), dtype='float32')
        padded_audio = np.concatenate((audio, silence), axis=0)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, padded_audio, SAMPLE_RATE)
            path = tmp.name

        try:
            # Transcribe speech
            segments, _ = model.transcribe(path, beam_size=1, vad_filter=True)
            for seg in segments:
                user_input = seg.text.strip()
                if user_input:
                    print(f"\n🗣️ You: {user_input}")
                    bot_reply = get_response_from_groq(user_input, chat_history)
                    print(f"🤖 Bot: {bot_reply}\n")
        except Exception as e:
            print("❌ Error during transcription:", e)
        finally:
            if os.path.exists(path):
                os.remove(path)

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
