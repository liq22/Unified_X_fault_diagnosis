#!/usr/bin/env python3
"""
Claude Code Response Capture Hook

Captures Claude's responses when conversation stops.
Triggered on Stop event.

Author: Claude Code Research System
Version: 1.0.0
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add cache system to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src" / "scripts" / "cache"))

try:
    from auto_hook import cache_thinking, cache_agent, get_simple_auto_hook, start_auto_cache
    from conversation_logger import log_claude_response, get_conversation_logger
    # Ensure auto-cache is started
    start_auto_cache()
    CACHE_AVAILABLE = True
    LOGGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cache system not available: {e}", file=sys.stderr)
    CACHE_AVAILABLE = False
    LOGGER_AVAILABLE = False

# Configure logging
log_file = project_root / "src" / "dev" / "cache" / "hooks.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('capture_response')

def read_transcript(transcript_path: str) -> dict:
    """Read conversation transcript from file"""
    try:
        if not transcript_path or not Path(transcript_path).exists():
            logger.warning(f"Transcript not found: {transcript_path}")
            return {}
            
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
            
        logger.info(f"Successfully read transcript: {len(transcript.get('messages', []))} messages")
        return transcript
        
    except Exception as e:
        logger.error(f"Failed to read transcript: {e}")
        return {}

def extract_claude_response(transcript: dict) -> str:
    """Extract Claude's latest response from transcript with full formatting"""
    try:
        messages = transcript.get('messages', [])
        if not messages:
            return ""
            
        # Get the last assistant message
        for message in reversed(messages):
            if message.get('role') == 'assistant':
                content = message.get('content', [])
                if isinstance(content, list):
                    # Extract text content while preserving formatting
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                            elif part.get('type') == 'tool_use':
                                # Include tool usage in the response
                                tool_name = part.get('name', 'unknown')
                                text_parts.append(f"\n🔧 Used tool: {tool_name}")
                        elif isinstance(part, str):
                            text_parts.append(part)
                    
                    # Join preserving line breaks and formatting
                    full_response = '\n'.join(text_parts)
                    
                    # Clean up excessive newlines but preserve intentional formatting
                    import re
                    full_response = re.sub(r'\n{3,}', '\n\n', full_response)
                    
                    return full_response
                elif isinstance(content, str):
                    return content
                    
        return ""
        
    except Exception as e:
        logger.error(f"Failed to extract Claude response: {e}")
        return ""

def extract_tool_usage(transcript: dict) -> list:
    """Extract tool usage from transcript"""
    try:
        messages = transcript.get('messages', [])
        tools_used = []
        
        for message in messages:
            if message.get('role') == 'assistant':
                content = message.get('content', [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'tool_use':
                            tools_used.append({
                                'name': part.get('name', ''),
                                'id': part.get('id', ''),
                                'input': part.get('input', {})
                            })
                            
        return tools_used
        
    except Exception as e:
        logger.error(f"Failed to extract tool usage: {e}")
        return []

def detect_thinking_content(response_text: str) -> str:
    """Extract thinking patterns from Claude's response"""
    try:
        # Look for common thinking patterns
        thinking_indicators = [
            "Let me think about",
            "I need to consider",
            "First, I'll",
            "My approach is",
            "The best way to",
            "I should",
            "Let me analyze"
        ]
        
        thinking_content = []
        lines = response_text.split('\n')
        
        for line in lines:
            for indicator in thinking_indicators:
                if indicator.lower() in line.lower():
                    thinking_content.append(line.strip())
                    break
                    
        return '\n'.join(thinking_content) if thinking_content else response_text[:500]
        
    except Exception as e:
        logger.error(f"Failed to detect thinking content: {e}")
        return response_text[:500] if response_text else "No thinking content detected"

def capture_claude_response(hook_data: dict):
    """Capture Claude's response from hook data"""
    try:
        # Extract key information
        session_id = hook_data.get('session_id', 'unknown')
        timestamp = datetime.now().isoformat()
        transcript_path = hook_data.get('transcript_path', '')
        
        # Load session data if available
        session_file = Path.home() / ".claude" / f"session_{session_id}.json"
        user_prompt = "Unknown prompt"
        
        try:
            if session_file.exists():
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    user_prompt = session_data.get('user_prompt', user_prompt)
        except Exception as e:
            logger.warning(f"Could not load session data: {e}")
        
        # Read transcript
        transcript = read_transcript(transcript_path)
        claude_response = extract_claude_response(transcript)
        tools_used = extract_tool_usage(transcript)
        thinking_content = detect_thinking_content(claude_response)
        
        # Log conversation in Markdown format
        if LOGGER_AVAILABLE and claude_response:
            try:
                log_claude_response(
                    session_id,
                    claude_response,
                    tools_used=[t['name'] for t in tools_used] if tools_used else None,
                    timestamp=timestamp
                )
                logger.info(f"Conversation logged to Markdown")
            except Exception as e:
                logger.error(f"Failed to log conversation: {e}")

        # Cache Claude's thinking and response
        if CACHE_AVAILABLE and claude_response:
            try:
                # Cache the thinking process with full content (no truncation)
                cache_path = cache_thinking(
                    user_prompt,
                    f"Claude response ({timestamp}):\n\nThinking: {thinking_content}\n\nFull response: {claude_response}",
                    ['response', 'claude_output', 'Stop', 'conversation']
                )
                
                if cache_path:
                    logger.info(f"Claude response cached: {Path(cache_path).name}")
                    print(f"🤖 Response cached: {Path(cache_path).name}", file=sys.stderr)
                
                # If tools were used, cache as agent execution
                if tools_used:
                    agent_name = f"claude-tools-{len(tools_used)}"
                    agent_cache_path = cache_agent(
                        agent_name,
                        {
                            'user_prompt': user_prompt,
                            'tools_used': tools_used,
                            'response': claude_response
                        },
                        [
                            {'step': 'tool_usage', 'tools': [t['name'] for t in tools_used]},
                            {'step': 'response_generation', 'timestamp': timestamp}
                        ],
                        {
                            'status': 'completed',
                            'tool_count': len(tools_used),
                            'response_length': len(claude_response)
                        }
                    )
                    
                    if agent_cache_path:
                        logger.info(f"Tool usage cached: {Path(agent_cache_path).name}")
                        print(f"🔧 Tools cached: {Path(agent_cache_path).name}", file=sys.stderr)
                        
            except Exception as e:
                logger.error(f"Cache error: {e}")
        
        # Clean up session file
        try:
            if session_file.exists():
                session_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean session file: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error capturing Claude response: {e}")
        return False

def main():
    """Main hook entry point"""
    try:
        # Read hook data from stdin
        hook_data = json.load(sys.stdin)
        logger.info(f"Received Stop hook data: {list(hook_data.keys())}")
        
        # Capture the response
        success = capture_claude_response(hook_data)
        
        if success:
            logger.info("Claude response successfully captured")
        else:
            logger.error("Failed to capture Claude response")
            
        # Return success (non-zero exit means hook failed)
        sys.exit(0)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON input: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hook execution error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()