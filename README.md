# Replica Unit (Rust)

A vibe-coded, local, always-listening. It scans the microphone with Whisper, recognizes the wake phrases **“Replica Unit”**, **“Replica”**, **“Unit”**, or **“Listen”**, records the request until a short pause, sends the transcription to a local Ollama model, and speaks the reply using Piper through the system's current audio output.

Whisper.cpp, Piper, Ollama, and the Qwen3.5 model run locally. No API key or cloud connection is needed. The suggested chat model is `qwen3.5:4b` (about 3.3–4.0 GB to download).

## Install prerequisites (Ubuntu / Debian)

```sh
sudo apt update
sudo apt install -y build-essential pkg-config libasound2-dev cmake git python3 python3-venv alsa-utils
```

Install Rust with [rustup](https://rustup.rs/).

## Set up speech recognition (Whisper.cpp)

Clone and build Whisper.cpp, then download the model:

```sh
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
sh ./models/download-ggml-model.sh base.en
cd ..
mkdir -p models
cp whisper.cpp/models/ggml-base.en.bin models/
```

## Set up the local LLM (Ollama)

Install Ollama using its [official instructions](https://ollama.com/download). Download the suggested model:

```sh
ollama pull qwen3.5:4b
```

Ollama normally starts as a background service on Linux. If it is not running, start it in a separate terminal with `ollama serve` and leave that terminal open. Ollama's local API defaults to `http://127.0.0.1:11434`.

## Set up speech output (Piper)

Create a Python virtual environment, install Piper, and download a voice and its config into this project's `models` directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install piper-tts
python3 -m piper.download_voices --data-dir models en_US-amy-medium
```

Piper downloads both `en_US-amy-medium.onnx` and its matching `.onnx.json` file. You can choose another voice from the [Piper voice catalog](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md).

## Configure and start the application

From the project directory, create the configuration file:

```sh
cp .env.example .env
```

Open `.env` and set the executable paths if `whisper-cli` and `piper` are not on your `PATH`. For the commands above, use:

```dotenv
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:4b
WHISPER_BIN=whisper.cpp/build/bin/whisper-cli
WHISPER_MODEL=models/ggml-base.en.bin
PIPER_BIN=.venv/bin/piper
PIPER_MODEL=models/en_US-amy-medium.onnx
```

Build and start the assistant:

```sh
cargo run --release
```

When the console says it is listening, say “Replica Unit”, “Replica”, “Unit”, or “Listen”, followed by your request. You can also say just the wake phrase, pause, and then say your request. The console prints the transcription, recognized request, and text sent to the local LLM. After the assistant replies aloud, it returns to listening. Press **Ctrl+C** to exit.

## Error reporting

Errors identify their component: `[audio input]`, `[STT]`, `[local LLM]`, or `[TTS/audio output]`. Missing Whisper or Piper model files stop startup and show the expected path. If Ollama is stopped or the selected model is unavailable, the app prints a local LLM error.

## Notes

The assistant reads microphone audio continuously and uses its measured level to find speech starts and ends. It transcribes each segment after microphone audio remains below the silence threshold continuously for 1.8 seconds; there is no fixed read or transcription interval. The assistant uses the default microphone and system audio output. The local model is loaded and run by Ollama; it may respond more slowly on computers with limited memory or without a supported GPU.

The `.gitignore` excludes the local `.env` configuration file and the root `models/` directory, so downloaded models and machine-specific settings are not included in Git.
