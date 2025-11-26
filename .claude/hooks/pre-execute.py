#!/usr/bin/env python3
"""
Claude Code Pre-Execute Hook

Automatically captures the start of every Claude Code execution.
This hook is called before any Claude Code command is executed.

Author: Claude Code Research System
Version: 1.0.0
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add scripts directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "scripts" / "logging"))
sys.path.append(str(project_root / "scripts" / "agent_management"))
sys.path.append(str(project_root / "src" / "scripts" / "cache"))

try:
    from claude_logger import start_logging_session, log_tool_usage
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Logging system not available: {e}", file=sys.stderr)
    LOGGING_AVAILABLE = False

try:
    from task_tracker import TaskTracker
    from goal_manager import GoalManager
    AGENT_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent management system not available: {e}", file=sys.stderr)
    AGENT_MANAGEMENT_AVAILABLE = False

try:
    from auto_hook import get_simple_auto_hook, cache_thinking
    CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cache system not available: {e}", file=sys.stderr)
    CACHE_AVAILABLE = False

def capture_execution_start(user_input: str = None, context: dict = None):
    """
    Capture the start of a Claude Code execution
    
    Args:
        user_input: The user's command/query
        context: Additional context information
    """
    session_id = None
    task_id = None
    agent_name = None
    
    try:
        # Extract user query from various sources
        user_query = None
        
        # Try to get from function parameters
        if user_input:
            user_query = user_input
        
        # Try to get from environment variables
        elif os.getenv('CLAUDE_USER_QUERY'):
            user_query = os.getenv('CLAUDE_USER_QUERY')
        
        # Try to get from command line arguments
        elif len(sys.argv) > 1:
            user_query = ' '.join(sys.argv[1:])
        
        # Default fallback
        else:
            user_query = "Claude Code execution session"
        
        # Start logging session if available
        if LOGGING_AVAILABLE:
            session_id = start_logging_session(user_query)
        
        # Cache the thinking process if available
        if CACHE_AVAILABLE and user_query:
            try:
                cache_path = cache_thinking(
                    user_query, 
                    "Pre-execution: User query captured", 
                    ["pre-execute-hook"]
                )
                if cache_path:
                    print(f"🧠 Thinking cached: {Path(cache_path).name}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to cache thinking: {e}", file=sys.stderr)
        
        # Detect agent usage and capture task if available
        if AGENT_MANAGEMENT_AVAILABLE:
            agent_name = detect_agent_usage(user_query)
            if agent_name:
                task_tracker = TaskTracker()
                task_id = task_tracker.capture_task(agent_name, user_query, context)
                
                # Cache agent execution start if available
                if CACHE_AVAILABLE:
                    try:
                        from auto_hook import cache_agent
                        agent_cache_path = cache_agent(
                            agent_name,
                            {"user_query": user_query, "context": context},
                            [{"step": "pre-execution", "timestamp": datetime.now().isoformat()}],
                            {"status": "started", "task_id": task_id}
                        )
                        if agent_cache_path:
                            print(f"🤖 Agent execution cached: {Path(agent_cache_path).name}", file=sys.stderr)
                    except Exception as e:
                        print(f"Warning: Failed to cache agent execution: {e}", file=sys.stderr)
        
        # Store session ID and task info for post-execute hook
        session_file = Path.home() / ".claude" / "current_session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        
        session_data = {
            'session_id': session_id,
            'task_id': task_id,
            'agent_name': agent_name,
            'start_time': datetime.now().isoformat(),
            'user_query': user_query,
            'pid': os.getpid(),
            'context': context or {}
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return session_id
        
    except Exception as e:
        print(f"Pre-execute hook error: {e}", file=sys.stderr)
        return None

def detect_agent_usage(user_query: str) -> str:
    """
    Detect which agent is being used based on the user query
    
    Args:
        user_query: The user's command/query
        
    Returns:
        Agent name if detected, None otherwise
    """
    try:
        # Look for /agent command pattern
        if "/agent" in user_query.lower():
            # Extract agent name from command
            parts = user_query.split()
            for i, part in enumerate(parts):
                if part.lower() == "/agent" and i + 1 < len(parts):
                    return parts[i + 1].split(':')[0]  # Remove any colon and parameters
        
        # Look for Task tool usage pattern (indicates agent invocation)
        if "Task tool" in user_query or "subagent" in user_query.lower():
            # Try to identify agent from context
            agent_keywords = {
                'research-literature': ['literature', 'papers', 'search', 'review'],
                'research-knowledge-graph': ['graph', 'network', 'citation', 'connections'],
                'research-hypothesis': ['hypothesis', 'theory', 'generate'],
                'research-gap-identifier': ['gap', 'opportunity', 'missing'],
                'research-trends': ['trend', 'future', 'prediction'],
                'research-academic': ['academic', 'scholar', 'database'],
                'research-semantic-scholar': ['semantic', 'scholar'],
                'writer-intro-cluster': ['introduction', 'intro', 'background'],
                'writer-method-cluster': ['method', 'methodology', 'algorithm'],
                'writer-results-cluster': ['results', 'experiment', 'data'],
                'writer-discussion-cluster': ['discussion', 'conclusion', 'analysis'],
                'writer-format-cluster': ['format', 'style', 'journal'],
                'writer-quality-controller': ['quality', 'review', 'validation'],
                'writer-style-formatter': ['format', 'style', 'nature', 'science'],
                'writer-cache-manager': ['cache', 'performance', 'optimization'],
                'coder-reviewer': ['code', 'review', 'security', 'quality'],
                'coder-debugger': ['debug', 'error', 'fix', 'troubleshoot'],
                'coder-industrial-ai': ['deploy', 'production', 'pytorch', 'jax']
            }
            
            query_lower = user_query.lower()
            for agent, keywords in agent_keywords.items():
                if any(keyword in query_lower for keyword in keywords):
                    return agent
        
        return None
        
    except Exception as e:
        print(f"Agent detection error: {e}", file=sys.stderr)
        return None

def main():
    """Main entry point for pre-execute hook"""
    # Get user input from command line or environment
    user_input = None
    
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
    
    # Additional context
    context = {
        'working_directory': os.getcwd(),
        'environment': dict(os.environ),
        'timestamp': datetime.now().isoformat()
    }
    
    session_id = capture_execution_start(user_input, context)
    
    # Provide feedback
    if session_id:
        print(f"📝 Logging session started: {session_id}", file=sys.stderr)
    
    # Show agent detection feedback if available
    if AGENT_MANAGEMENT_AVAILABLE and user_input:
        agent_name = detect_agent_usage(user_input)
        if agent_name:
            print(f"🎯 Goal-oriented execution: {agent_name}", file=sys.stderr)
        else:
            print(f"💭 General execution (no specific agent detected)", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())