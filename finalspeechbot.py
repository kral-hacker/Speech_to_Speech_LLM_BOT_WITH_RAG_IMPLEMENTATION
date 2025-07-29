import queue
import sounddevice as sd
import numpy as np
import webrtcvad
import collections
import openai
import os
import soundfile as sf
import tempfile
import threading
import time
import pyttsx3
import signal
import sys
from faster_whisper import WhisperModel

# === CONFIG ===
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)
NUM_FRAMES = int(1000 / FRAME_DURATION)  # Increased from 400 to 1000ms for better stability
DEVICE = "cuda" if os.environ.get("USE_GPU") else "cpu"
MIN_AUDIO_LENGTH = 0.5  # Minimum audio length in seconds before processing

# === Setup Groq API ===
openai.api_key = "##"
openai.api_base = "https://api.groq.com/openai/v1"

# === Whisper Model ===
model = WhisperModel("base.en", device=DEVICE, compute_type="int8")

# === VAD and Queues ===
vad = webrtcvad.Vad(1)  # Changed from 2 to 1 for less sensitive detection
audio_queue = queue.Queue()
response_queue = queue.Queue()

# === Chat History ===
chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

# === TTS Setup ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)
tts_queue = queue.Queue()
speaking = threading.Event()

# === Exit Flag ===
exit_flag = threading.Event()

# === TTS Thread ===
def tts_worker():
    while not exit_flag.is_set():
        try:
            text = tts_queue.get(timeout=1)
            if text is None:
                break
            
            try:
                speaking.set()
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ TTS Error: {e}")
            finally:
                speaking.clear()  # Always clear the speaking flag
                
        except queue.Empty:
            continue

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# === Get Groq Response ===
def get_response_from_groq(prompt, history):
    # Create a copy of history to avoid modifying the original during API call
    temp_history = history.copy()
    temp_history.append({"role": "user", "content": prompt})
    
    try:
        response = openai.ChatCompletion.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=temp_history,
            temperature=0.7,
            max_tokens=512
        )
        reply = response.choices[0].message["content"]
        
        # Only update history after successful API call
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        
        return reply
    except Exception as e:
        print(f"❌ Error contacting Groq API: {e}")
        return f"Sorry, I'm having trouble connecting to the AI service right now."

# === Speak Function ===
def speak_text(text):
    if not exit_flag.is_set():
        tts_queue.put(text)

# === Audio Callback ===
def callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}")
    
    if not exit_flag.is_set():
        # Convert CFFI buffer to numpy array first, then to int16 bytes
        audio_np = np.frombuffer(indata, dtype=np.int16)
        audio_data = audio_np.tobytes()
        audio_queue.put(audio_data)

# === VAD Collector Thread ===
def vad_collector():
    ring_buffer = collections.deque(maxlen=NUM_FRAMES)
    triggered = False
    voiced_frames = []

    while not exit_flag.is_set():
        try:
            frame = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        if speaking.is_set():
            continue  # Don't process audio while speaking

        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception as e:
            print(f"VAD error: {e}")
            continue

        if not triggered:
            ring_buffer.append(frame)
            speech_frames = sum(1 for f in ring_buffer if vad.is_speech(f, SAMPLE_RATE))
            # Increased threshold from 0.9 to 0.95 for more confidence
            if speech_frames > 0.95 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
                print("🎙️ Speech detected, recording...")
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            speech_frames = sum(1 for f in ring_buffer if vad.is_speech(f, SAMPLE_RATE))
            # Decreased threshold from 0.1 to 0.05 for longer silence requirement
            if speech_frames < 0.05 * ring_buffer.maxlen:
                triggered = False
                audio_data = b''.join(voiced_frames)
                
                # Check minimum audio length
                audio_length = len(audio_data) / (SAMPLE_RATE * 2)  # 2 bytes per sample
                if audio_length >= MIN_AUDIO_LENGTH:
                    print(f"🔊 Audio captured ({audio_length:.1f}s), processing...")
                    response_queue.put(audio_data)
                else:
                    print(f"⏭️ Audio too short ({audio_length:.1f}s), ignoring...")
                
                ring_buffer.clear()
                voiced_frames.clear()

# === Transcribe Thread ===
def transcribe_loop():
    while not exit_flag.is_set():
        try:
            audio_data = response_queue.get(timeout=1)
        except queue.Empty:
            continue

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name
                with sf.SoundFile(temp_file, mode='w', samplerate=SAMPLE_RATE, channels=1, subtype='PCM_16') as f:
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    f.write(audio_np.reshape(-1, 1))

            # Add a small delay to ensure file is fully written
            time.sleep(0.1)
            
            segments, _ = model.transcribe(temp_file, beam_size=1, vad_filter=True)
            for seg in segments:
                user_input = seg.text.strip()
                # Filter out very short or meaningless transcriptions
                if user_input and len(user_input) > 2 and user_input not in ['.', '..', '...', ',', '!', '?'] and not exit_flag.is_set():
                    print(f"\n🗣️ You: {user_input}")
                    bot_reply = get_response_from_groq(user_input, chat_history)
                    print(f"🤖 Bot: {bot_reply}\n")
                    speak_text(bot_reply)
                    
                    # Wait for speaking to finish with timeout
                    timeout = 30  # 30 second timeout for TTS
                    start_time = time.time()
                    while speaking.is_set() and not exit_flag.is_set():
                        if time.time() - start_time > timeout:
                            print("⚠️ TTS timeout, continuing...")
                            break
                        time.sleep(0.1)
                    
                    print("🎤 Ready for next input...")
                elif user_input:
                    print(f"⏭️ Ignoring short/noise transcription: '{user_input}'")
                        
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
        finally:
            if temp_file:
                try:
                    time.sleep(0.1)  # Small delay before deletion
                    os.remove(temp_file)
                except (PermissionError, OSError) as e:
                    print(f"⚠️ Could not remove temp file {temp_file}: {e}")

# === Signal Handler ===
def signal_handler(sig, frame):
    print("\n🛑 Exiting... Cleaning up.")
    exit_flag.set()
    
    # Stop TTS
    tts_queue.put(None)
    
    # Give threads time to finish
    time.sleep(1)
    
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Main Entry ===
def main():
    print("🟢 Voice Assistant is running. Start speaking!")
    
    try:
        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=FRAME_SIZE, 
            dtype=np.int16,  # Use numpy dtype instead of string
            channels=1, 
            callback=callback
        )
        
        with stream:
            # Start worker threads
            vad_thread = threading.Thread(target=vad_collector, daemon=True)
            transcribe_thread = threading.Thread(target=transcribe_loop, daemon=True)
            
            vad_thread.start()
            transcribe_thread.start()
            
            print("🎤 Listening... Press Ctrl+C to stop.")
            
            # Main loop
            while not exit_flag.is_set():
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received. Exiting...")
        signal_handler(None, None)
    except Exception as e:
        print(f"❌ Error starting audio stream: {e}")
        exit_flag.set()

if __name__ == "__main__":
    main()