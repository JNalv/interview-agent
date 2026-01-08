"""Transcript generation and export."""
from pathlib import Path
from datetime import datetime
from typing import Optional
from .anthropic_client import InterviewClient

def cleanup_transcript(
    raw_transcript: str,
    client: InterviewClient
) -> str:
    """
    Clean up transcript using Claude API.
    
    Args:
        raw_transcript: Raw transcript with potential errors
        client: InterviewClient instance
        
    Returns:
        Cleaned transcript
    """
    return client.cleanup_transcript(raw_transcript)

def export_transcript(
    transcript: str,
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None
) -> Path:
    """
    Export transcript to .txt file.
    
    Args:
        transcript: Final transcript text
        output_dir: Output directory (defaults to current dir)
        filename: Custom filename (defaults to timestamped name)
        
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_transcript_{timestamp}.txt"
    
    if not filename.endswith(".txt"):
        filename += ".txt"
    
    file_path = output_dir / filename
    
    # Format transcript with header
    formatted = f"INTERVIEW TRANSCRIPT\n"
    formatted += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formatted += f"\n{'='*50}\n\n"
    formatted += transcript
    formatted += f"\n\n{'='*50}\n"
    formatted += "[End of Interview]"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(formatted)
    
    return file_path

