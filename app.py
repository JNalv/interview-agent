"""Main Gradio application for Interview Agent."""
import gradio as gr
from pathlib import Path
from typing import Optional, Tuple, List
import os
from modules import (
    load_documents, Transcriber, InterviewClient, 
    ContextManager, export_transcript, cleanup_transcript
)
from config import (
    DEFAULT_PROMPT_PATH, TOKEN_WARNING_THRESHOLD,
    TOKEN_CRITICAL_THRESHOLD, MAX_CONTEXT_TOKENS
)

# Global state
transcriber: Optional[Transcriber] = None
client: Optional[InterviewClient] = None
context: Optional[ContextManager] = None
current_answer_text: str = ""


def load_default_prompt() -> str:
    """Load default system prompt from file."""
    if DEFAULT_PROMPT_PATH.exists():
        with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_documents_handler(folder_path: str) -> Tuple[str, str]:
    """
    Handle document loading.
    
    Returns:
        Tuple of (status_message, status_message) for doc_status
    """
    global context
    
    if not folder_path or not folder_path.strip():
        return "Please provide a folder path.", "Please provide a folder path."
    
    try:
        text, file_count, errors = load_documents(folder_path)
        
        if file_count == 0:
            return "No supported documents found in folder.", "No supported documents found in folder."
        
        # Initialize context if needed
        if context is None:
            context = ContextManager()
        
        context.set_documents(text)
        
        # Load default prompt if not set
        if not context.system_prompt:
            prompt = load_default_prompt()
            if prompt:
                context.set_system_prompt(prompt)
        
        error_msg = ""
        if errors:
            error_msg = f" ({len(errors)} files had errors)"
        
        status = f"✓ Loaded {file_count} document(s){error_msg}. Ready to start interview."
        return status, status
    except Exception as e:
        error_msg = f"Error loading documents: {str(e)}"
        return error_msg, error_msg


def show_prompt_editor() -> Tuple[gr.Group, str]:
    """Show the prompt editor modal."""
    prompt_text = load_default_prompt()
    if context and context.system_prompt:
        prompt_text = context.system_prompt
    return gr.Group(visible=True), prompt_text


def hide_prompt_editor() -> gr.Group:
    """Hide the prompt editor modal."""
    return gr.Group(visible=False)


def save_prompt_handler(prompt_text: str) -> Tuple[gr.Group, str]:
    """
    Save the system prompt.
    
    Returns:
        Tuple of (modal_visibility, status_message)
    """
    global context
    
    if not prompt_text or not prompt_text.strip():
        return gr.Group(visible=True), "Prompt cannot be empty."
    
    if context is None:
        context = ContextManager()
    
    context.set_system_prompt(prompt_text.strip())
    return gr.Group(visible=False), "System prompt saved."


def start_interview() -> Tuple[List, bool, str]:
    """
    Start the interview by getting the first question.
    
    Returns:
        Tuple of (conversation_messages, interview_active, status_message)
    """
    global client, context
    
    if context is None or not context.document_text:
        return [], False, "Please load documents first."
    
    if not context.system_prompt:
        return [], False, "Please set a system prompt first."
    
    try:
        # Initialize client if needed
        if client is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return [], False, "Error: ANTHROPIC_API_KEY environment variable not set."
            client = InterviewClient(api_key=api_key)
        
        # Get first question - Anthropic API requires at least one message
        # Send an initial prompt to start the interview
        messages = [{"role": "user", "content": "Please begin the interview by introducing yourself and asking your first question based on the provided context documents."}]
        system_prompt = context.get_full_system_prompt()
        
        response = client.send_message(system_prompt, messages, max_tokens=1024)
        
        # Add first question as a turn (with empty answer for now)
        context.add_turn(response, "")
        
        # Return conversation format (Gradio Chatbot expects list of dicts with 'role' and 'content')
        conversation_messages = [{"role": "assistant", "content": response}]
        return conversation_messages, True, "✓ Interview started! Click 'Start Recording' to answer."
    except Exception as e:
        return [], False, f"Error starting interview: {str(e)}"


def start_recording_handler(interview_active: bool) -> Tuple[str, bool]:
    """
    Start recording audio.
    
    Returns:
        Tuple of (status_message, recording_state)
    """
    global transcriber
    
    if not interview_active:
        return "Please start the interview first.", False
    
    try:
        if transcriber is None:
            transcriber = Transcriber(use_gpu=True)
        
        transcriber.start_recording()
        return "🎤 Recording... Speak now.", True
    except Exception as e:
        return f"Error starting recording: {str(e)}", False


def stop_recording_handler(is_recording: bool, interview_active: bool) -> Tuple[str, List, bool, str]:
    """
    Stop recording and get transcription, then get next question.
    
    Returns:
        Tuple of (transcription, conversation_messages, recording_state, status_message)
    """
    global transcriber, client, context, current_answer_text
    
    if not is_recording:
        return "", [], False, ""
    
    if not interview_active or context is None:
        return "", [], False, "Error: Interview not active or context not initialized."
    
    try:
        # Stop recording and get transcription
        transcription = transcriber.stop_recording()
        current_answer_text = transcription
        
        if not transcription.strip():
            return "", [], False, "No speech detected. Please try again."
        
        # Get the last question (which should have empty answer)
        if not context.turns:
            return transcription, [], False, "Error: No questions in context."
        
        # Update the last turn with the answer
        last_turn = context.turns[-1]
        last_turn.answer = transcription
        last_turn.answer_tokens = client.count_tokens(transcription)
        
        # Get next question from Claude
        messages = context.get_messages()
        system_prompt = context.get_full_system_prompt()
        
        next_question = client.send_message(system_prompt, messages, max_tokens=1024)
        
        # Add new turn with next question
        context.add_turn(next_question, "")
        
        # Build conversation display - include all completed turns plus the new question
        # Gradio Chatbot expects list of dicts with 'role' and 'content' keys
        conversation_messages = []
        for turn in context.turns:
            conversation_messages.append({"role": "assistant", "content": turn.question})
            if turn.answer:
                conversation_messages.append({"role": "user", "content": turn.answer})
        
        return transcription, conversation_messages, False, "✓ Answer recorded. Next question ready."
    except Exception as e:
        error_msg = f"Error processing answer: {str(e)}"
        return transcription if 'transcription' in locals() else "", [], False, error_msg


def end_interview_handler() -> Tuple[str, str, bool, Optional[str]]:
    """
    End the interview and generate transcript.
    
    Returns:
        Tuple of (status_message, transcript_file_path, interview_active, file_path)
    """
    global context, client, transcriber
    
    if context is None or not context.turns:
        return "No interview to end.", None, False, None
    
    try:
        # Stop recording if active
        if transcriber and transcriber._is_recording:
            transcriber.stop_recording()
        
        # Get raw transcript
        raw_transcript = context.get_raw_transcript()
        
        # Clean up transcript using Claude
        if client:
            cleaned_transcript = cleanup_transcript(raw_transcript, client)
        else:
            cleaned_transcript = raw_transcript
        
        # Export to file
        file_path = export_transcript(cleaned_transcript)
        
        status = f"Interview ended. Transcript saved to: {file_path}"
        return status, str(file_path), False, str(file_path)
    except Exception as e:
        return f"Error ending interview: {str(e)}", None, False, None


def update_token_display() -> Tuple[float, str, float]:
    """
    Update token usage display.
    
    Returns:
        Tuple of (token_percentage, token_info_text, turns_remaining)
    """
    if context is None:
        return 0.0, "No documents loaded", 0.0
    
    try:
        used_tokens, max_tokens, percentage = context.get_token_usage()
        turns_remaining = context.get_turns_remaining()
        
        # Format status message with warnings
        if percentage >= TOKEN_CRITICAL_THRESHOLD * 100:
            status = f"🔴 Near limit! {used_tokens:,} tokens ({percentage:.1f}%) - Consider ending soon"
        elif percentage >= TOKEN_WARNING_THRESHOLD * 100:
            status = f"⚠️ Approaching limit: {used_tokens:,} tokens ({percentage:.1f}%)"
        else:
            status = f"{used_tokens:,} tokens used ({percentage:.1f}%)"
        
        return percentage, status, float(turns_remaining)
    except Exception as e:
        return 0.0, f"Error calculating tokens: {str(e)}", 0.0


def create_ui() -> gr.Blocks:
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Interview Agent") as app:
        # State
        interview_active = gr.State(False)
        is_recording = gr.State(False)
        
        gr.Markdown("# Interview Agent")
        gr.Markdown("Load your context documents, then start the interview. Use voice recording to answer questions.")
        
        with gr.Row():
            # Left column: Controls
            with gr.Column(scale=1):
                # Document upload section
                gr.Markdown("### Setup")
                doc_folder_picker = gr.File(
                    label="📁 Select Document Folder",
                    file_count="directory"
                )
                doc_status = gr.Textbox(label="Document Status", interactive=False, value="No documents loaded")
                
                gr.Markdown("---")
                
                # System prompt editor
                gr.Markdown("### System Prompt")
                edit_prompt_btn = gr.Button("Edit System Prompt")
                prompt_status = gr.Textbox(label="Prompt Status", interactive=False, value="", visible=False)
                
                gr.Markdown("---")
                
                # Interview controls
                gr.Markdown("### Interview Controls")
                start_interview_btn = gr.Button("Start Interview", variant="primary")
                interview_status = gr.Textbox(label="Status", interactive=False, value="", visible=True)
                
                # Recording controls
                with gr.Row():
                    record_btn = gr.Button("🎤 Start Recording", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Recording", variant="secondary")
                
                # Interview control
                end_interview_btn = gr.Button("End Interview", variant="stop")
                
                gr.Markdown("---")
                
                # Token usage display
                gr.Markdown("### Token Usage")
                token_bar = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Context Usage (%)", interactive=False
                )
                token_info = gr.Textbox(
                    label="Status", 
                    value="No documents loaded",
                    interactive=False
                )
                turns_remaining = gr.Number(
                    label="Estimated Turns Remaining",
                    value=0, interactive=False
                )
            
            # Right column: Conversation
            with gr.Column(scale=2):
                gr.Markdown("### Interview Conversation")
                conversation = gr.Chatbot(
                    label="Interview",
                    height=500,
                    value=[]
                )
                current_answer = gr.Textbox(
                    label="Your Current Answer (transcription)",
                    lines=3,
                    interactive=False,
                    value=""
                )
        
        # System prompt editor modal
        with gr.Group(visible=False) as prompt_modal:
            gr.Markdown("### Edit System Prompt")
            prompt_editor = gr.Textbox(
                label="System Prompt",
                lines=15,
                value=load_default_prompt(),
                autofocus=False
            )
            with gr.Row():
                save_prompt_btn = gr.Button("Save Prompt", variant="primary")
                cancel_prompt_btn = gr.Button("Cancel")
        
        # Download component for transcript
        transcript_file = gr.File(label="Download Transcript", visible=False)
        
        # Event handlers
        def extract_folder_path(files):
            """Extract folder path from file picker selection."""
            if not files:
                return ""
            
            try:
                # Gradio File component with file_count="directory" returns a list of files
                # When a directory is selected, it returns all files within that directory
                # We need to find the common parent directory of all files
                if isinstance(files, list) and len(files) > 0:
                    file_paths = []
                    for f in files:
                        file_str = f if isinstance(f, str) else (f.name if hasattr(f, 'name') else str(f))
                        file_paths.append(Path(file_str))
                    
                    # Find common parent directory
                    if len(file_paths) == 1:
                        # Single file - get its parent
                        folder_path = file_paths[0].parent
                    else:
                        # Multiple files - find common parent
                        common_parts = []
                        for parts in zip(*[p.parts for p in file_paths]):
                            if len(set(parts)) == 1:
                                common_parts.append(parts[0])
                            else:
                                break
                        folder_path = Path(*common_parts) if common_parts else file_paths[0].parent
                    
                    # Verify it's a directory
                    if folder_path.is_dir():
                        return str(folder_path.resolve())
                    else:
                        # Fallback: try parent of first file
                        return str(file_paths[0].parent.resolve())
                elif isinstance(files, str):
                    file_path = Path(files)
                    if file_path.is_dir():
                        return str(file_path.resolve())
                    else:
                        return str(file_path.parent.resolve())
            except Exception as e:
                print(f"Error extracting folder path: {e}")
                import traceback
                traceback.print_exc()
                return ""
            return ""
        
        def load_documents_from_picker(files):
            """Load documents directly from folder picker."""
            folder_path = extract_folder_path(files)
            if not folder_path:
                return "Please select a folder.", "Please select a folder."
            print(f"Loading documents from folder: {folder_path}")
            return load_documents_handler(folder_path)
        
        doc_folder_picker.change(
            fn=load_documents_from_picker,
            inputs=[doc_folder_picker],
            outputs=[doc_status, doc_status]
        ).then(
            fn=update_token_display,
            inputs=[],
            outputs=[token_bar, token_info, turns_remaining]
        )
        
        edit_prompt_btn.click(
            fn=show_prompt_editor,
            inputs=[],
            outputs=[prompt_modal, prompt_editor]
        )
        
        cancel_prompt_btn.click(
            fn=hide_prompt_editor,
            inputs=[],
            outputs=[prompt_modal]
        )
        
        save_prompt_btn.click(
            fn=save_prompt_handler,
            inputs=[prompt_editor],
            outputs=[prompt_modal, prompt_status]
        ).then(
            fn=lambda: gr.Textbox(visible=True),
            inputs=[],
            outputs=[prompt_status]
        )
        
        start_interview_btn.click(
            fn=start_interview,
            inputs=[],
            outputs=[conversation, interview_active, interview_status]
        ).then(
            fn=update_token_display,
            inputs=[],
            outputs=[token_bar, token_info, turns_remaining]
        )
        
        record_btn.click(
            fn=start_recording_handler,
            inputs=[interview_active],
            outputs=[interview_status, is_recording]
        )
        
        stop_btn.click(
            fn=stop_recording_handler,
            inputs=[is_recording, interview_active],
            outputs=[current_answer, conversation, is_recording, interview_status]
        ).then(
            fn=update_token_display,
            inputs=[],
            outputs=[token_bar, token_info, turns_remaining]
        )
        
        end_interview_btn.click(
            fn=end_interview_handler,
            inputs=[],
            outputs=[interview_status, transcript_file, interview_active, transcript_file]
        ).then(
            fn=lambda f: gr.File(visible=True) if f else gr.File(visible=False),
            inputs=[transcript_file],
            outputs=[transcript_file]
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())

