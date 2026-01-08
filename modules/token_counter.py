"""Token counting utilities."""
import tiktoken
from config import MAX_CONTEXT_TOKENS, OVERHEAD_BUFFER_TOKENS, AVG_TOKENS_PER_TURN

# Use cl100k_base encoding (Claude-compatible approximation)
_encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """
    Count tokens in text.
    
    Args:
        text: Text to count
        
    Returns:
        Token count
    """
    return len(_encoding.encode(text))

def estimate_turns_remaining(
    used_tokens: int,
    max_tokens: int = MAX_CONTEXT_TOKENS,
    overhead: int = OVERHEAD_BUFFER_TOKENS,
    avg_per_turn: int = AVG_TOKENS_PER_TURN
) -> int:
    """
    Estimate remaining conversation turns.
    
    Args:
        used_tokens: Currently used tokens
        max_tokens: Maximum context window
        overhead: Safety buffer
        avg_per_turn: Average tokens per Q&A turn
        
    Returns:
        Estimated remaining turns
    """
    remaining = max_tokens - used_tokens - overhead
    return max(0, remaining // avg_per_turn)

