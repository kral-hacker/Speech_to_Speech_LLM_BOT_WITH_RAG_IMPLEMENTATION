import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import tempfile
import os
import threading
import queue
import sys
import time
import webrtcvad
from gtts import gTTS
import pygame
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from groq import Groq

# === CONFIG ===
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 10  # VAD frame duration
SILENCE_THRESHOLD_SECONDS = 0.5  # For faster recording termination
VAD_SENSITIVITY = 3  # VAD aggressiveness (0-3)
DEVICE = "cuda" if os.environ.get("USE_GPU") else "cpu"
SILENCE_PADDING = 1.0  # Padding after speech ends
TTS_START_DELAY = 0.2  # Short delay for interruption detection
INTERRUPT_CHECK_DURATION = 0.05  # Fast VAD check
PRE_PLAYBACK_DELAY = 0.1  # Ensure playback initialization
RECORD_TIMEOUT = 10.0  # Timeout to prevent recording hang

# === Setup Groq API ===
client = Groq(api_key="##")

# === Load Whisper Model ===
model = WhisperModel("base.en", device=DEVICE, compute_type="int8")

# === Initialize LangChain RAG ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None  # Initialize as None, will create on first write
documents = []  # Store documents for FAISS

# === Conversation History ===
chat_history = [{"role": "system", "content": "You are HelpBuddy, a helpful assistant created by Ananay Tyagi. When answering questions, use relevant context from stored documents if available."}]

# === Shared state ===
interrupted = threading.Event()
tts_queue = queue.Queue()
tts_lock = threading.Lock()
speaking = threading.Event()  # Track if TTS is currently speaking

# Initialize pygame mixer with mono settings
pygame.mixer.pre_init(16000, -16, 1, 1024)
pygame.mixer.init()

def store_in_vector_store(text):
    global vector_store, documents
    print(f"Storing in vector store: {text[:50]}...")  # Debug
    doc = Document(page_content=text)
    documents.append(doc)
    if vector_store is None:
        vector_store = FAISS.from_documents([doc], embeddings)
    else:
        vector_store.add_documents([doc])
    print("Document stored successfully.")  # Debug

def retrieve_relevant_context(query):
    global vector_store
    if vector_store is None or not documents:
        return None
    print(f"Retrieving context for query: {query[:50]}...")  # Debug
    results = vector_store.similarity_search(query, k=1)
    if results:
        return results[0].page_content
    return None

def get_response_from_groq(prompt, history):
    # Check if the prompt is a "write" command
    if prompt.lower().startswith("remember the following information"):
        content_to_write = prompt[6:].strip()
        if content_to_write:
            store_in_vector_store(content_to_write)
            history.append({"role": "user", "content": prompt})
            reply = "Content has been written to the knowledge base."
            history.append({"role": "assistant", "content": reply})
            return reply
        else:
            return "No content provided to write."

    # For questions, retrieve relevant context
    context = retrieve_relevant_context(prompt)
    augmented_prompt = prompt
    if context:
        augmented_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        print(f"Augmented prompt with context: {augmented_prompt[:100]}...")  # Debug

    history.append({"role": "user", "content": prompt})
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history + [{"role": "user", "content": augmented_prompt}],
            temperature=0.7,
            max_tokens=512
        )
        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return f"[Error contacting Groq API: {e}]"

def tts_worker():
    while True:
        try:
            text = tts_queue.get(timeout=0.5)
            if text is None:
                print("TTS worker received stop signal.")
                break
            print(f"TTS worker processing text: {text[:50]}...")  # Debug
            with tts_lock:
                if not interrupted.is_set():
                    speaking.set()
                    try:
                        tts = gTTS(text=text, lang='en', slow=False)
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=tempfile.gettempdir()) as tmp:
                            tts.save(tmp.name)
                            tmp_path = tmp.name
                        print(f"Generated MP3 file: {tmp_path}")  # Debug
                        try:
                            sound = pygame.mixer.Sound(tmp_path)
                            time.sleep(PRE_PLAYBACK_DELAY)
                            channel = sound.play()
                            print(f"Playing audio: {tmp_path}")  # Debug
                            if channel is None:
                                print("⚠️ Playback failed: No channel assigned")
                                pygame.mixer.quit()
                                pygame.mixer.init()
                                sound = pygame.mixer.Sound(tmp_path)
                                channel = sound.play()
                                print(f"Retrying audio: {tmp_path}")
                            start_time = time.time()
                            while channel and channel.get_busy() and not interrupted.is_set():
                                time.sleep(0.05)
                            duration = time.time() - start_time
                            print(f"Audio playback duration: {duration:.2f} seconds")  # Debug
                        finally:
                            sound.stop()
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                                    print(f"Cleaned up MP3 file: {tmp_path}")  # Debug
                            except OSError as e:
                                print(f"⚠️ File cleanup error: {e}")
                    except Exception as err:
                        print(f"⚠️ TTS Error: {err}")
                    finally:
                        tts_queue.task_done()
        except queue.Empty:
            continue
        except Exception as err:
            print(f"⚠️ TTS worker error: {err}")
        finally:
            speaking.clear()

def detect_interrupt():
    duration = INTERRUPT_CHECK_DURATION
    vad = webrtcvad.Vad(VAD_SENSITIVITY)
    try:
        print(f"Waiting {TTS_START_DELAY} seconds before interrupt detection")  # Debug
        time.sleep(TTS_START_DELAY)
        while not interrupted.is_set() and speaking.is_set():
            audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            # Convert float32 audio to int16 for webrtcvad
            audio_int16 = (audio * 32768).astype(np.int16).tobytes()
            is_speech = vad.is_speech(audio_int16, SAMPLE_RATE)
            energy = np.linalg.norm(audio)
            print(f"Interrupt check: is_speech={is_speech}, energy={energy:.6f}")  # Debug
            if is_speech:
                interrupted.set()
                with tts_lock:
                    pygame.mixer.stop()
                    while not tts_queue.empty():
                        try:
                            tts_queue.get_nowait()
                            tts_queue.task_done()
                        except queue.Empty:
                            break
                print("\n\N{WARNING SIGN} Bot interrupted by user.\n")
                break
            time.sleep(0.01)  # Short delay to prevent CPU overload
    except Exception as e:
        print(f"⚠️ Interrupt detection error: {e}")
    finally:
        interrupted.clear()  # Ensure flag is reset

def record_with_vad():
    audio_buffer = []
    silence_duration = 0
    is_speaking = False
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=FRAME_SIZE)
    stream.start()
    start_time = time.time()

    try:
        while time.time() - start_time < RECORD_TIMEOUT:
            frame, overflowed = stream.read(FRAME_SIZE)
            if overflowed:
                continue
            frame_int16 = (frame * 32768).astype(np.int16).tobytes()
            is_speech = vad.is_speech(frame_int16, SAMPLE_RATE)
            print(f"VAD check: is_speech={is_speech}, energy={np.linalg.norm(frame):.6f}")  # Debug
            if is_speech:
                is_speaking = True
                silence_duration = 0
                audio_buffer.append(frame)
            elif is_speaking:
                silence_duration += FRAME_DURATION_MS / 1000.0
                audio_buffer.append(frame)
                if silence_duration >= SILENCE_THRESHOLD_SECONDS:
                    break
            else:
                continue
    finally:
        stream.stop()
        stream.close()

    if not audio_buffer:
        print("No speech detected in recording")  # Debug
        return None

    audio = np.concatenate(audio_buffer, axis=0)
    silence = np.zeros((int(SILENCE_PADDING * SAMPLE_RATE), 1), dtype='float32')
    padded_audio = np.concatenate((audio, silence), axis=0)
    return padded_audio

# === VAD Setup ===
vad = webrtcvad.Vad(VAD_SENSITIVITY)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # Samples per frame

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

try:
    print("\N{STUDIO MICROPHONE} Speak to the bot. Say 'write' followed by content to store it, or ask a question to retrieve relevant information. Press Ctrl+C to exit.")

    while True:
        # Stop any ongoing playback and clear queue
        interrupted.set()
        with tts_lock:
            pygame.mixer.stop()
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except queue.Empty:
                    break
        interrupted.clear()

        # Record audio with VAD
        print("Starting new recording...")  # Debug
        audio = record_with_vad()
        if audio is None:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, SAMPLE_RATE)
            path = tmp.name

        try:
            segments, _ = model.transcribe(path, beam_size=1, vad_filter=False)
            for seg in segments:
                user_input = seg.text.strip()
                if user_input and len(user_input.split()) >= 2:
                    print(f"\n\N{SPEECH BALLOON} You: {user_input}")
                    bot_reply = get_response_from_groq(user_input, chat_history)
                    print(f"\N{ROBOT FACE} Bot: {bot_reply}\n")

                    # === Handle TTS + Interruption ===
                    interrupted.clear()

                    # Add new text to queue
                    print(f"Queueing TTS: {bot_reply[:50]}...")  # Debug
                    tts_queue.put(bot_reply)

                    # Start interrupt detector
                    interrupt_thread = threading.Thread(target=detect_interrupt, daemon=True)
                    interrupt_thread.start()

                    # Wait for TTS to finish or be interrupted
                    while speaking.is_set() and not interrupted.is_set():
                        time.sleep(0.05)

                    if interrupted.is_set():
                        speaking.clear()
                        print("Interrupt cleanup completed.")  # Debug
                        time.sleep(0.1)  # Brief delay to release audio device

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
        finally:
            if os.path.exists(path):
                os.remove(path)

except KeyboardInterrupt:
    print("\n🛑 Stopped.")
except Exception as e:
    print(f"❌ Unexpected error: {e}", file=sys.stderr)
finally:
    tts_queue.put(None)  # Stop the TTS thread
    pygame.mixer.quit()