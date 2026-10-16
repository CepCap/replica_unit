use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde_json::json;
use std::{env, fs, io::Write, path::{Path, PathBuf}, process::{Command, Stdio}, sync::mpsc, time::{Duration, SystemTime, UNIX_EPOCH}};

const TARGET_RATE: u32 = 16_000;
const FORCED_LANGUAGE: &str = "en";
const WAKE_WORDS: &[&str] = &["example"];
const SILENCE_RMS_THRESHOLD: f32 = 0.012;
const END_OF_UTTERANCE_SILENCE_MS: usize = 1_800;

enum AudioEvent {
    Samples(Vec<f32>),
    Error(String),
}

struct Config {
    ollama_url: String,
    ollama_model: String,
    whisper_bin: String,
    whisper_model: PathBuf,
    piper_bin: String,
    piper_model: PathBuf,
    sample_rate: u32,
    channels: u16,
}

fn config() -> Result<Config> {
    Ok(Config {
        ollama_url: env::var("OLLAMA_URL").context("OLLAMA_URL must be set")?,
        ollama_model: env::var("OLLAMA_MODEL").context("OLLAMA_MODEL must be set")?,
        whisper_bin: env::var("WHISPER_BIN").context("WHISPER_BIN must be set")?,
        whisper_model: env::var("WHISPER_MODEL").context("WHISPER_MODEL must be set")?.into(),
        piper_bin: env::var("PIPER_BIN").context("PIPER_BIN must be set")?,
        piper_model: env::var("PIPER_MODEL").context("PIPER_MODEL must be set")?.into(),
        sample_rate: TARGET_RATE,
        channels: 1,
    })
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let mut cfg = config()?;
    println!("Replica Unit — English-only; listening for “Replica Unit”, “Replica”, “Unit”, or “Listen”. Ctrl+C exits.");
    check_file(&cfg.whisper_model, "STT", "Whisper model")?;
    check_file(&cfg.piper_model, "TTS", "Piper voice model")?;

    let host = cpal::default_host();
    let input = host.default_input_device().ok_or_else(|| anyhow!("ERROR [audio input]: No default microphone is available."))?;
    let input_name = input.name().unwrap_or_else(|_| "unknown microphone".into());
    let supported = input.default_input_config().context("ERROR [audio input]: Could not read microphone configuration")?;
    let sample_format = supported.sample_format();
    let stream_config: cpal::StreamConfig = supported.into();
    cfg.sample_rate = stream_config.sample_rate.0;
    cfg.channels = stream_config.channels;
    println!("Microphone: {input_name} ({} Hz, {} channels)", cfg.sample_rate, cfg.channels);

    let (audio_tx, audio_rx) = mpsc::channel::<AudioEvent>();
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let samples_tx = audio_tx.clone();
            let errors_tx = audio_tx.clone();
            input.build_input_stream(&stream_config, move |d: &[f32], _| { let _ = samples_tx.send(AudioEvent::Samples(d.to_vec())); }, move |e| { let _ = errors_tx.send(AudioEvent::Error(e.to_string())); }, None)
        }
        cpal::SampleFormat::I16 => {
            let samples_tx = audio_tx.clone();
            let errors_tx = audio_tx.clone();
            input.build_input_stream(&stream_config, move |d: &[i16], _| { let _ = samples_tx.send(AudioEvent::Samples(d.iter().map(|x| *x as f32 / i16::MAX as f32).collect())); }, move |e| { let _ = errors_tx.send(AudioEvent::Error(e.to_string())); }, None)
        }
        cpal::SampleFormat::U16 => {
            let samples_tx = audio_tx.clone();
            let errors_tx = audio_tx.clone();
            input.build_input_stream(&stream_config, move |d: &[u16], _| { let _ = samples_tx.send(AudioEvent::Samples(d.iter().map(|x| (*x as f32 - 32768.0) / 32768.0).collect())); }, move |e| { let _ = errors_tx.send(AudioEvent::Error(e.to_string())); }, None)
        }
        other => bail!("ERROR [audio input]: Unsupported microphone sample format: {other:?}"),
    }.context("ERROR [audio input]: Could not open microphone stream")?;
    stream.play().context("ERROR [audio input]: Could not start microphone stream")?;

    let mut segment = Vec::<f32>::new();
    let mut speech_active = false;
    let mut silent_samples = 0usize;
    let mut awaiting_command = false;
    let required_silence = cfg.sample_rate as usize * END_OF_UTTERANCE_SILENCE_MS / 1_000;
    loop {
        let block = match audio_rx.recv() {
            Ok(AudioEvent::Samples(samples)) => samples,
            Ok(AudioEvent::Error(error)) => bail!("ERROR [audio input]: {error}"),
            Err(_) => bail!("ERROR [audio input]: Microphone stream stopped."),
        };
        let mono = to_mono(&block, cfg.channels as usize);
        let voiced = rms(&mono) > SILENCE_RMS_THRESHOLD;

        if !speech_active {
            if !voiced {
                continue;
            }
            speech_active = true;
            segment.clear();
            silent_samples = 0;
            println!("Speech detected; recording until silence.");
        }

        segment.extend(&mono);
        if voiced {
            silent_samples = 0;
        } else {
            silent_samples += mono.len();
        }

        if silent_samples >= required_silence {
            speech_active = false;
            silent_samples = 0;
            let completed_segment = std::mem::take(&mut segment);
            process_segment(&cfg, &completed_segment, &mut awaiting_command)?;
        }
    }
}

fn process_segment(cfg: &Config, audio: &[f32], awaiting_command: &mut bool) -> Result<()> {
    let decoded = match transcribe(cfg, audio) {
        Ok(text) => text.trim().to_owned(),
        Err(error) => {
            eprintln!("ERROR [STT]: {error:#}");
            return Ok(());
        }
    };
    if decoded.is_empty() {
        eprintln!("ERROR [STT]: No speech could be recognized.");
        return Ok(());
    }
    println!("\nDecoded audio: {decoded}");

    if *awaiting_command {
        *awaiting_command = false;
        let request = strip_wake_words(&decoded);
        if request.is_empty() {
            eprintln!("ERROR [STT]: No request was recognized after the wake phrase.");
            return Ok(());
        }
        return send_request(cfg, &request);
    }

    if !is_wake_word(&decoded) {
        return Ok(());
    }

    println!("Wake phrase recognized.");
    let request = strip_wake_words(&decoded);
    if request.is_empty() {
        println!("Wake phrase heard. Listening for your request until silence.");
        *awaiting_command = true;
        return Ok(());
    }
    send_request(cfg, &request)
}

fn send_request(cfg: &Config, text: &str) -> Result<()> {
    println!("Recognized after wake phrase: {text}");
    println!("Sent to local LLM ({}): {text}", cfg.ollama_model);
    let response = match chat(cfg, text) {
        Ok(response) => response,
        Err(error) => {
            eprintln!("ERROR [local LLM]: {error:#}");
            return Ok(());
        }
    };
    println!("Replica Unit: {response}");
    if let Err(error) = speak(cfg, &response) {
        eprintln!("ERROR [TTS/audio output]: {error:#}");
    }
    Ok(())
}

fn transcribe(cfg: &Config, audio: &[f32]) -> Result<String> {
    let wav = temp_path("input.wav");
    write_wav(&wav, audio, cfg.sample_rate, cfg.channels)?;
    let output = Command::new(&cfg.whisper_bin).args(["-m", cfg.whisper_model.to_string_lossy().as_ref(), "-f", wav.to_string_lossy().as_ref(), "-l", FORCED_LANGUAGE, "-nt", "-np"]).output()
        .with_context(|| format!("Unable to start '{}'. Install whisper.cpp and set WHISPER_BIN.", cfg.whisper_bin))?;
    let _ = fs::remove_file(&wav);
    if !output.status.success() { bail!("whisper.cpp failed: {}", String::from_utf8_lossy(&output.stderr).trim()); }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if text.is_empty() { bail!("whisper.cpp returned empty transcription"); }
    Ok(text)
}

fn chat(cfg: &Config, prompt: &str) -> Result<String> {
    let client = reqwest::blocking::Client::builder().timeout(Duration::from_secs(300)).build()?;
    let url = format!("{}/api/chat", cfg.ollama_url.trim_end_matches('/'));
    let personality = "You are Replica Unit, a helpful voice assistant. Do not use asterisks in your response. Try to keep answers short, below 5 sentences";
    let response = client.post(url)
        .json(&json!({
            "model": cfg.ollama_model,
            "stream": false,
            "think": false,
            "messages": [
                {"role":"system","content":personality},
                {"role":"user","content":prompt}
            ]
        }))
        .send()
        .context("Could not connect to Ollama. Make sure Ollama is running locally.")?;
    let status = response.status();
    let body: serde_json::Value = response.json().context("Invalid response from Ollama")?;
    if !status.is_success() {
        bail!("Ollama returned {status}: {}", body["error"].as_str().unwrap_or("unknown local LLM error"));
    }
    body["message"]["content"].as_str().map(str::to_owned)
        .ok_or_else(|| anyhow!("Ollama response did not contain assistant text"))
}

fn speak(cfg: &Config, text: &str) -> Result<()> {
    let wav = temp_path("reply.wav");
    let mut child = Command::new(&cfg.piper_bin).args(["--model", cfg.piper_model.to_string_lossy().as_ref(), "--output_file", wav.to_string_lossy().as_ref()]).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn()
        .with_context(|| format!("Unable to start '{}'. Install Piper and set PIPER_BIN.", cfg.piper_bin))?;
    child.stdin.take().ok_or_else(|| anyhow!("Could not open Piper input"))?.write_all(text.as_bytes()).context("Could not send text to Piper")?;
    let result = child.wait_with_output()?;
    if !result.status.success() { bail!("Piper failed: {}", String::from_utf8_lossy(&result.stderr).trim()); }
    if !wav.exists() { bail!("Piper did not create audio output"); }
    let player = ["paplay", "aplay", "ffplay"].into_iter().find(|p| Command::new(p).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok());
    let player = player.ok_or_else(|| anyhow!("No audio player found. Install paplay, aplay, or ffplay."))?;
    let status = match player {
        "paplay" => Command::new(player).arg(&wav).status(),
        "aplay" => Command::new(player).arg(&wav).status(),
        _ => Command::new(player).args(["-nodisp", "-autoexit", "-loglevel", "error"]).arg(&wav).status(),
    }.context("Could not play assistant audio")?;
    let _ = fs::remove_file(wav);
    if !status.success() { bail!("Audio player exited with {status}"); }
    Ok(())
}

fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 { return samples.to_vec(); }
    samples.chunks(channels).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect()
}
fn rms(samples: &[f32]) -> f32 { (samples.iter().map(|x| x*x).sum::<f32>() / samples.len().max(1) as f32).sqrt() }
fn is_wake_word(text: &str) -> bool {
    let normalized = text.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' }).collect::<String>();
    WAKE_WORDS.iter().any(|word| normalized.split_whitespace().collect::<Vec<_>>().join(" ").contains(word))
}
fn temp_path(name: &str) -> PathBuf {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    env::temp_dir().join(format!("replica-unit-{stamp}-{name}"))
}
fn write_wav(path: &Path, data: &[f32], rate: u32, _channels: u16) -> Result<()> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(path, spec).context("Could not create temporary WAV for STT")?;
    for sample in data { writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?; }
    writer.finalize()?;
    Ok(())
}
fn check_file(path: &Path, component: &str, label: &str) -> Result<()> {
    if !path.is_file() { bail!("ERROR [{component}]: {label} not found at '{}'. Follow the setup instructions in README.md.", path.display()); }
    Ok(())
}

fn strip_wake_words(text: &str) -> String {
    let leading = text.trim_start();
    let lower = leading.to_lowercase();
    for phrase in WAKE_WORDS {
        if lower.starts_with(phrase) {
            let end = phrase.len();
            let tail = &leading[end..];
            if tail.chars().next().map(|c| c.is_whitespace() || ",.!?—-:".contains(c)).unwrap_or(true) {
                let cleaned = tail.trim().trim_start_matches(|c: char| ",.!?—-: ".contains(c)).trim().to_owned();
                if !cleaned.is_empty() { return cleaned; }
            }
        }
    }
    text.to_owned()
}
