#!/usr/bin/env python3
"""
Claude Code Prompt Capture Hook

Captures user prompts when submitted to Claude Code.
Triggered on UserPromptSubmit event.

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
    from auto_hook import cache_thinking, cache_research, get_simple_auto_hook, start_auto_cache
    from conversation_logger import log_user_prompt, get_conversation_logger
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
logger = logging.getLogger('capture_prompt')

def capture_user_prompt(hook_data: dict):
    """Capture user prompt from hook data"""
    try:
        # Extract key information from hook data
        session_id = hook_data.get('session_id', 'unknown')
        timestamp = datetime.now().isoformat()
        prompt_text = hook_data.get('prompt', '')
        
        # Enhanced prompt data
        prompt_content = {
            'type': 'user_prompt',
            'session_id': session_id,
            'prompt': prompt_text,
            'timestamp': timestamp,
            'hook_event': 'UserPromptSubmit',
            'cwd': hook_data.get('cwd', ''),
            'transcript_path': hook_data.get('transcript_path', '')
        }
        
        # Log prompt in Markdown format
        if LOGGER_AVAILABLE and prompt_text:
            try:
                log_user_prompt(session_id, prompt_text, timestamp)
                logger.info(f"User prompt logged to Markdown")
            except Exception as e:
                logger.error(f"Failed to log prompt: {e}")

        # Cache the prompt
        if CACHE_AVAILABLE:
            try:
                cache_path = cache_thinking(
                    prompt_text,
                    f"User prompt captured at {timestamp}",
                    ['prompt', 'user_input', 'UserPromptSubmit']
                )
                
                if cache_path:
                    logger.info(f"User prompt cached: {Path(cache_path).name}")
                    print(f"📝 Prompt cached: {Path(cache_path).name}", file=sys.stderr)
                else:
                    logger.warning("Failed to cache user prompt")
                    
            except Exception as e:
                logger.error(f"Cache error: {e}")
                
        # Store session context for other hooks
        session_file = Path.home() / ".claude" / f"session_{session_id}.json"
        session_data = {
            'session_id': session_id,
            'start_time': timestamp,
            'user_prompt': prompt_text,
            'transcript_path': hook_data.get('transcript_path', ''),
            'cwd': hook_data.get('cwd', '')
        }
        
        try:
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            logger.info(f"Session data saved: {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error capturing user prompt: {e}")
        return False

def main():
    """Main hook entry point"""
    try:
        # Read hook data from stdin
        hook_data = json.load(sys.stdin)
        logger.info(f"Received UserPromptSubmit hook data: {list(hook_data.keys())}")
        
        # Capture the prompt
        success = capture_user_prompt(hook_data)
        
        if success:
            logger.info("User prompt successfully captured")
        else:
            logger.error("Failed to capture user prompt")
            
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