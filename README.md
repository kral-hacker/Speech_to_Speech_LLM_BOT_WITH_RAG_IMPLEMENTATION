# 🗣️ Speech-to-Speech LLM Bot with RAG Integration

This project is a real-time **speech-based assistant** powered by **LLMs**, **FasterWhisper ASR**, **text-to-speech (TTS)**, **LangChain RAG**, and **Groq API**. You can talk to the bot, and it will respond naturally — including retrieving relevant documents using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- 🔊 Real-time **speech recognition** with [FasterWhisper](https://github.com/guillaumekln/faster-whisper)
- 🤖 Natural language responses powered by **Groq LLM API**
- 📚 Document-aware answers with **LangChain + FAISS-based RAG**
- 🗣️ **Speech output** using **gTTS + pygame**
- 🧠 Memory-like behavior — remembers what you say if instructed
- 🧵 Fully interruptible TTS playback via threading

---

## 🧾 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 📁 Folder Structure
```bash
├── speech_bot_with_rag.py          # 🔹 Main entry point
├── finalspeechbot.py               # Older version
├── speech_bot_interruptable.py     # Standalone voice bot
├── speech_to_mistral.py            # For Mistral-specific testing
├── vectorstore/                    # FAISS vector DB
├── docs/                           # Your uploaded or processed documents
├── .env                            # Environment file with API keys
└── requirements.txt
```

## ▶️ How to Run
```bash
python speech_bot_with_rag.py
```

## 🎤 Example Conversations
You: “Hey, how are you?”
Bot: “I’m doing well, how can I assist you today?”

You: “Remember this — I have a meeting at 8 PM.”
Bot: “Got it. I’ll remember that you have a meeting at 8 PM.”

You: “Do I have any meetings today?”
Bot: “Yes, you have a meeting scheduled at 8 PM.”

## 🧠 RAG: Retrieval-Augmented Generation
The bot indexes your uploaded documents using sentence-transformers and FAISS. It retrieves relevant text chunks during conversation and feeds them to the LLM to generate context-aware responses.

To add new documents:

Drop them into the docs/ folder.

The script will automatically update the vectorstore.




