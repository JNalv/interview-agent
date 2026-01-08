"""Manage conversation context and history."""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from .token_counter import count_tokens, estimate_turns_remaining
from config import MAX_CONTEXT_TOKENS

@dataclass
class Turn:
    """Single Q&A turn in the interview."""
    question: str
    answer: str
    question_tokens: int = 0
    answer_tokens: int = 0

@dataclass
class ContextManager:
    """Manages full interview context including documents and history."""
    
    system_prompt: str = ""
    document_text: str = ""
    turns: List[Turn] = field(default_factory=list)
    
    _system_tokens: int = 0
    _document_tokens: int = 0
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt and cache token count."""
        self.system_prompt = prompt
        self._system_tokens = count_tokens(prompt)
    
    def set_documents(self, text: str) -> None:
        """Set document context and cache token count."""
        self.document_text = text
        self._document_tokens = count_tokens(text)
    
    def add_turn(self, question: str, answer: str) -> None:
        """Add a Q&A turn to history."""
        turn = Turn(
            question=question,
            answer=answer,
            question_tokens=count_tokens(question),
            answer_tokens=count_tokens(answer)
        )
        self.turns.append(turn)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get conversation in Claude message format.
        
        Returns:
            List of message dicts with 'role' and 'content'
        """
        messages = []
        for turn in self.turns:
            messages.append({"role": "assistant", "content": turn.question})
            messages.append({"role": "user", "content": turn.answer})
        return messages
    
    def get_full_system_prompt(self) -> str:
        """
        Get combined system prompt with document context.
        
        Returns:
            System prompt with documents appended
        """
        if self.document_text:
            return f"{self.system_prompt}\n\nContext Documents:\n{self.document_text}"
        return self.system_prompt
    
    def get_token_usage(self) -> Tuple[int, int, float]:
        """
        Get current token usage stats.
        
        Returns:
            Tuple of (used_tokens, max_tokens, percentage)
        """
        # Count tokens in all turns
        turn_tokens = sum(
            turn.question_tokens + turn.answer_tokens 
            for turn in self.turns
        )
        
        # Total includes system prompt, documents, and conversation
        used_tokens = self._system_tokens + self._document_tokens + turn_tokens
        percentage = (used_tokens / MAX_CONTEXT_TOKENS) * 100
        
        return used_tokens, MAX_CONTEXT_TOKENS, percentage
    
    def get_turns_remaining(self) -> int:
        """Estimate remaining turns before context limit."""
        used_tokens, _, _ = self.get_token_usage()
        return estimate_turns_remaining(used_tokens)
    
    def get_raw_transcript(self) -> str:
        """Get raw Q&A transcript for export."""
        lines = []
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"Question {i}:")
            lines.append(turn.question)
            lines.append("")
            lines.append(f"Answer {i}:")
            lines.append(turn.answer)
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

