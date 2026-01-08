"""Configuration constants for Interview Agent."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "default_system_prompt.txt"

# Anthropic API
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
MAX_CONTEXT_TOKENS = 200_000
TOKEN_WARNING_THRESHOLD = 0.75  # 75% - yellow warning
TOKEN_CRITICAL_THRESHOLD = 0.90  # 90% - red warning
OVERHEAD_BUFFER_TOKENS = 5_000
AVG_TOKENS_PER_TURN = 600  # ~100 question + ~500 answer

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Whisper model settings (faster-whisper)
# Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
# "base" is a good balance of speed and quality
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cuda"  # "cuda" for GPU, "cpu" for CPU
WHISPER_COMPUTE_TYPE = "float16"  # "float16" for GPU, "int8" for CPU

# Supported document formats
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

