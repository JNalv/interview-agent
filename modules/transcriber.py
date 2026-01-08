"""faster-whisper speech-to-text transcriber with CUDA support."""
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional
import threading
import tempfile
import wave
from config import SAMPLE_RATE, CHANNELS, WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")


class Transcriber:
    """Speech transcription using faster-whisper (record then transcribe)."""
    
    def __init__(self, model_size: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize transcriber with faster-whisper model.
        
        Args:
            model_size: Whisper model size (e.g., "base", "small", "medium", "large-v3")
            use_gpu: Whether to use CUDA acceleration
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("faster-whisper is not installed. Install with: pip install faster-whisper")
        
        self._model_size = model_size or WHISPER_MODEL_SIZE
        self._device = WHISPER_DEVICE if use_gpu else "cpu"
        self._compute_type = WHISPER_COMPUTE_TYPE if use_gpu else "int8"
        
        # Initialize Whisper model
        print(f"Loading Whisper model: {self._model_size} on {self._device}...")
        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type
            )
            print(f"Whisper model loaded successfully on {self._device}")
        except Exception as e:
            # Fallback to CPU if GPU fails
            if use_gpu and self._device == "cuda":
                print(f"Warning: GPU initialization failed ({e}), falling back to CPU")
                self._device = "cpu"
                self._compute_type = "int8"
                self._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8"
                )
            else:
                raise RuntimeError(f"Failed to initialize Whisper model: {e}") from e
        
        # Audio recording state
        self._audio_buffer: list[np.ndarray] = []
        self._is_recording: bool = False
        self._audio_stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
    
    def start_recording(self) -> None:
        """Begin audio capture from microphone."""
        if self._is_recording:
            return
        
        with self._lock:
            self._is_recording = True
            self._audio_buffer = []
        
        # Start audio capture stream
        self._audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=1024
        )
        self._audio_stream.start()
    
    def stop_recording(self) -> str:
        """
        Stop recording and return transcribed text.
        
        Returns:
            Transcribed text from audio
        """
        if not self._is_recording:
            return ""
        
        # Stop audio stream
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None
        
        # Get audio data
        with self._lock:
            self._is_recording = False
            if not self._audio_buffer:
                return ""
            
            # Concatenate all audio chunks
            audio_data = np.concatenate(self._audio_buffer)
            self._audio_buffer = []
        
        # Transcribe using faster-whisper
        try:
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Write WAV file
            with wave.open(tmp_path, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                # Convert float32 [-1, 1] to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Transcribe
            segments, info = self._model.transcribe(
                tmp_path,
                beam_size=5,
                language="en"
            )
            
            # Collect transcription text
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text.strip())
            
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
            
            transcription = " ".join(transcription_parts).strip()
            return transcription
            
        except Exception as e:
            # Clean up temp file on error
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                        time_info: dict, status: sd.CallbackFlags) -> None:
        """Sounddevice callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")
        
        if self._is_recording:
            # Convert to mono if stereo
            if indata.shape[1] > 1:
                audio_data = np.mean(indata, axis=1)
            else:
                audio_data = indata[:, 0]
            
            # Add to buffer
            with self._lock:
                if self._is_recording:
                    self._audio_buffer.append(audio_data.copy())
