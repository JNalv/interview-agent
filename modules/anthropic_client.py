"""Anthropic API client for interview interactions."""
import anthropic
import os
from typing import List, Dict
from .token_counter import count_tokens
from config import ANTHROPIC_MODEL

class InterviewClient:
    """Wrapper for Anthropic API with interview-specific methods."""
    
    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
        """
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._client = anthropic.Anthropic(api_key=api_key)
    
    def send_message(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024
    ) -> str:
        """
        Send message to Claude and get response.
        
        Args:
            system_prompt: System prompt for interview persona
            messages: Conversation history in Claude format
            max_tokens: Maximum tokens in response
            
        Returns:
            Claude's response text
        """
        try:
            response = self._client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )
            return response.content[0].text
        except anthropic.APIError as e:
            raise RuntimeError(f"Anthropic API error: {e.message}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Anthropic API: {str(e)}") from e
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken approximation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Approximate token count
        """
        return count_tokens(text)
    
    def cleanup_transcript(self, raw_transcript: str) -> str:
        """
        Send transcript to Claude for cleanup and error correction.
        
        Args:
            raw_transcript: Raw Q&A transcript with potential transcription errors
            
        Returns:
            Cleaned transcript
        """
        cleanup_prompt = """You are a transcript editor. Review the following interview transcript and:
1. Fix any transcription errors (homophones, misheard words, filler words)
2. Format it as a clean Q&A transcript
3. Preserve the meaning and intent of all responses
4. Remove excessive filler words like "um", "uh", "like" but keep natural flow
5. Fix any obvious grammar errors while maintaining the speaker's voice

Return the cleaned transcript in the same format."""

        messages = [
            {"role": "user", "content": f"Please clean up this transcript:\n\n{raw_transcript}"}
        ]
        
        try:
            cleaned = self.send_message(cleanup_prompt, messages, max_tokens=8000)
            return cleaned
        except Exception as e:
            # If cleanup fails, return original transcript
            print(f"Warning: Transcript cleanup failed: {e}")
            return raw_transcript

