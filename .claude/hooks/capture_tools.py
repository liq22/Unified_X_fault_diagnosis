#!/usr/bin/env python3
"""
Claude Code Tool Usage Capture Hook

Captures tool usage when Claude uses tools.
Triggered on PostToolUse event.

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
    # Ensure auto-cache is started
    start_auto_cache()
    CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cache system not available: {e}", file=sys.stderr)
    CACHE_AVAILABLE = False

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
logger = logging.getLogger('capture_tools')

def capture_tool_usage(hook_data: dict):
    """Capture tool usage from hook data"""
    try:
        # Extract key information
        session_id = hook_data.get('session_id', 'unknown')
        timestamp = datetime.now().isoformat()
        tool_name = hook_data.get('tool_name', 'unknown_tool')
        tool_input = hook_data.get('tool_input', {})
        tool_output = hook_data.get('tool_output', {})
        
        # Create tool execution summary
        tool_summary = {
            'tool_name': tool_name,
            'timestamp': timestamp,
            'session_id': session_id,
            'input_params': tool_input,
            'output_data': tool_output,
            'hook_event': 'PostToolUse'
        }
        
        # Generate user-friendly description
        tool_description = f"Tool '{tool_name}' executed at {timestamp}"
        if isinstance(tool_input, dict):
            if 'file_path' in tool_input:
                tool_description += f" on file: {tool_input['file_path']}"
            elif 'command' in tool_input:
                tool_description += f" with command: {tool_input['command'][:100]}..."
            elif 'pattern' in tool_input:
                tool_description += f" searching for: {tool_input['pattern']}"
                
        # Cache tool usage as agent execution
        if CACHE_AVAILABLE:
            try:
                # Use tool name as agent name
                agent_name = f"tool-{tool_name.lower().replace(' ', '-')}"
                
                # Prepare detailed tool execution data
                execution_steps = [
                    {
                        'step': 'tool_invocation',
                        'tool': tool_name,
                        'timestamp': timestamp,
                        'input_size': len(str(tool_input))
                    }
                ]
                
                # Add specific steps based on tool type
                if tool_name in ['Read', 'Write', 'Edit']:
                    execution_steps.append({
                        'step': 'file_operation',
                        'file_path': tool_input.get('file_path', ''),
                        'operation': tool_name.lower()
                    })
                elif tool_name in ['Bash', 'BashOutput']:
                    execution_steps.append({
                        'step': 'command_execution',
                        'command': tool_input.get('command', '')[:200]
                    })
                elif tool_name in ['Grep', 'Glob']:
                    execution_steps.append({
                        'step': 'search_operation',
                        'pattern': tool_input.get('pattern', ''),
                        'path': tool_input.get('path', '')
                    })
                elif tool_name == 'Task':
                    execution_steps.append({
                        'step': 'agent_invocation',
                        'subagent': tool_input.get('subagent_type', ''),
                        'description': tool_input.get('description', '')
                    })
                
                # Determine execution result
                execution_result = {
                    'status': 'completed',
                    'tool_name': tool_name,
                    'execution_time': timestamp,
                    'success': not ('error' in str(tool_output).lower()),
                    'output_size': len(str(tool_output))
                }
                
                # Add specific results based on tool output
                if isinstance(tool_output, dict):
                    if 'content' in tool_output:
                        execution_result['content_length'] = len(str(tool_output['content']))
                    if 'files' in tool_output:
                        execution_result['files_found'] = len(tool_output['files'])
                    if 'matches' in tool_output:
                        execution_result['matches_found'] = len(tool_output['matches'])
                
                # Cache the tool execution
                cache_path = cache_agent(
                    agent_name,
                    {
                        'tool_name': tool_name,
                        'input_parameters': tool_input,
                        'execution_context': hook_data.get('cwd', ''),
                        'session_id': session_id
                    },
                    execution_steps,
                    execution_result
                )
                
                if cache_path:
                    logger.info(f"Tool usage cached: {tool_name} -> {Path(cache_path).name}")
                    print(f"🔧 Tool cached: {tool_name}", file=sys.stderr)
                
                # Also cache as thinking process for tool reasoning
                thinking_content = f"Tool execution: {tool_name}\n"
                thinking_content += f"Context: {tool_description}\n"
                thinking_content += f"Input: {json.dumps(tool_input, indent=2)[:300]}...\n"
                if isinstance(tool_output, dict) and 'content' in tool_output:
                    thinking_content += f"Output preview: {str(tool_output['content'])[:200]}...\n"
                
                thinking_cache_path = cache_thinking(
                    f"Tool usage: {tool_name}",
                    thinking_content,
                    ['tool_usage', f'tool_{tool_name.lower()}', 'PostToolUse']
                )
                
                if thinking_cache_path:
                    logger.info(f"Tool thinking cached: {Path(thinking_cache_path).name}")
                    
            except Exception as e:
                logger.error(f"Cache error: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error capturing tool usage: {e}")
        return False

def main():
    """Main hook entry point"""
    try:
        # Read hook data from stdin
        hook_data = json.load(sys.stdin)
        logger.info(f"Received PostToolUse hook data: {hook_data.get('tool_name', 'unknown')} tool")
        
        # Capture the tool usage
        success = capture_tool_usage(hook_data)
        
        if success:
            logger.info(f"Tool usage successfully captured: {hook_data.get('tool_name', 'unknown')}")
        else:
            logger.error(f"Failed to capture tool usage: {hook_data.get('tool_name', 'unknown')}")
            
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