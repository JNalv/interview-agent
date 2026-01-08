# Interview Agent

An intelligent interview application that uses GPU-accelerated speech-to-text transcription (faster-whisper) and Claude Sonnet 4.5 for conducting insightful interviews based on document context.

## Features

- **Document Context Loading**: Load and process documents (.txt, .md, .pdf, .docx) as interview context
- **GPU-Accelerated Transcription**: High-quality speech-to-text using faster-whisper (Whisper models) with automatic CUDA support
- **Intelligent Question Generation**: Claude Sonnet 4.5 generates contextual, insightful questions
- **Token Management**: Real-time token usage tracking with warnings and turn estimation
- **Customizable System Prompts**: Edit the interviewer persona before starting
- **Transcript Export**: Generate cleaned, formatted transcripts saved as .txt files

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Ubuntu/Debian Linux
- Anthropic API key

## Installation

1. Clone or navigate to the project directory:
```bash
cd /home/jnalv/llm-projects/interview-agent
```

2. Install system dependencies (required for audio recording):
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

3. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

   This will install `faster-whisper` which includes CUDA support automatically if your system has CUDA installed.

5. Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser to `http://localhost:7860`

3. **Load Documents**: Enter the path to a folder containing your context documents and click "Load Documents"

4. **Edit System Prompt** (optional): Click "Edit System Prompt" to customize the interviewer persona

5. **Start Interview**: Click "Start Interview" to begin. The AI will ask the first question based on your documents.

6. **Answer Questions**:
   - Click "🎤 Start Recording" to begin speaking
   - Click "⏹️ Stop Recording" when finished
   - Your answer will be transcribed and sent to Claude for the next question

7. **End Interview**: Click "End Interview" to generate and download the cleaned transcript

## Token Limits

- **Context Window**: 200,000 tokens (Claude Sonnet 4.5)
- **Warnings**: 
  - Yellow warning at 75% capacity
  - Red warning at 90% capacity
- **Turn Estimation**: Displays estimated remaining Q&A turns based on average token usage

## Supported Document Formats

- `.txt` - Plain text files
- `.md` - Markdown files
- `.pdf` - PDF documents
- `.docx` - Microsoft Word documents

## Troubleshooting

### Whisper Model Download
The Whisper model will be automatically downloaded on first use by faster-whisper. Models are cached in your system's cache directory.

You can change the model size in `config.py`:
- `"tiny"` - Fastest, lowest quality
- `"base"` - Good balance (default)
- `"small"` - Better quality, slower
- `"medium"` - High quality, slower
- `"large-v3"` - Best quality, slowest

### CUDA/GPU Support
faster-whisper automatically detects and uses CUDA if available. If CUDA isn't available, it will automatically fall back to CPU mode.

**To force CPU mode:** Edit `config.py` and set:
```python
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
```

**Note:** GPU acceleration requires CUDA and cuDNN to be installed. If you see CUDA errors, the application will automatically use CPU mode instead.

### PortAudio Library Not Found
If you see `OSError: PortAudio library not found`, install the system dependency:
```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio
```

### Microphone Not Working
Ensure your microphone is properly configured and accessible. The application uses `sounddevice` for audio capture, which requires PortAudio to be installed.

## Project Structure

```
interview-agent/
├── app.py                    # Main Gradio application
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── modules/
│   ├── document_loader.py    # Document loading and parsing
│   ├── transcriber.py        # faster-whisper STT wrapper
│   ├── anthropic_client.py   # Claude API client
│   ├── context_manager.py    # Conversation context management
│   ├── token_counter.py      # Token counting utilities
│   └── transcript_export.py  # Transcript generation and export
├── prompts/
│   └── default_system_prompt.txt
```

## License

This project is provided as-is for personal use.

