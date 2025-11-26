#!/usr/bin/env python3
"""
Claude Code Post-Execute Hook

Automatically captures the completion of every Claude Code execution.
This hook is called after any Claude Code command is completed.

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
    from claude_logger import end_logging_session, log_response, log_tool_usage, log_agent_invocation, log_file_access
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Logging system not available: {e}", file=sys.stderr)
    LOGGING_AVAILABLE = False

try:
    from task_tracker import TaskTracker
    from goal_manager import GoalManager
    from summary_generator import SummaryGenerator
    AGENT_MANAGEMENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Agent management system not available: {e}", file=sys.stderr)
    AGENT_MANAGEMENT_AVAILABLE = False

try:
    from auto_hook import get_simple_auto_hook, cache_thinking, cache_agent, cache_research
    CACHE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Cache system not available: {e}", file=sys.stderr)
    CACHE_AVAILABLE = False

def capture_execution_end(success: bool = True, error_message: str = None, response: str = None, context: dict = None):
    """
    Capture the end of a Claude Code execution
    
    Args:
        success: Whether the execution was successful
        error_message: Error message if execution failed
        response: Claude's response text
        context: Additional context information
    """
    log_filename = None
    summary_id = None
    
    try:
        # Load current session data
        session_file = Path.home() / ".claude" / "current_session.json"
        
        if not session_file.exists():
            print("Warning: No active session found", file=sys.stderr)
            return None
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        session_id = session_data.get('session_id')
        task_id = session_data.get('task_id')
        agent_name = session_data.get('agent_name')
        start_time = session_data.get('start_time')
        user_query = session_data.get('user_query', '')
        
        # Calculate execution time
        execution_time = 0.0
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
                execution_time = (datetime.now() - start_dt).total_seconds()
            except:
                pass
        
        # Log execution metadata if logging is available
        if LOGGING_AVAILABLE and session_id:
            # Log Claude's response if provided
            if response:
                log_response(response)
            
            # Log execution metadata from environment or context
            if context:
                # Log tools used
                tools_used = context.get('tools_used', [])
                for tool in tools_used:
                    log_tool_usage(
                        tool_name=tool.get('name', ''), 
                        parameters=tool.get('parameters', {}),
                        duration=tool.get('duration', 0)
                    )
                
                # Log agents invoked
                agents_used = context.get('agents_used', [])
                for agent in agents_used:
                    log_agent_invocation(
                        agent_name=agent.get('name', ''),
                        parameters=agent.get('parameters', {}),
                        duration=agent.get('duration', 0)
                    )
                
                # Log files accessed
                files_accessed = context.get('files_accessed', [])
                for file_access in files_accessed:
                    log_file_access(
                        file_path=file_access.get('path', ''),
                        operation=file_access.get('operation', 'unknown')
                    )
            
            # End the logging session
            log_filename = end_logging_session(success, error_message)
        
        # Process agent management tasks if available
        if AGENT_MANAGEMENT_AVAILABLE and agent_name and task_id:
            try:
                # Calculate performance metrics
                performance_metrics = calculate_performance_metrics(success, execution_time, context)
                
                # Update task completion
                task_tracker = TaskTracker()
                task_tracker.update_task_completion(
                    task_id, 
                    performance_metrics, 
                    execution_time, 
                    generate_completion_summary(success, agent_name, performance_metrics, error_message)
                )
                
                # Update goal manager with performance metrics
                goal_manager = GoalManager()
                goal_manager.update_agent_performance(
                    agent_name, 
                    performance_metrics, 
                    user_query, 
                    execution_time
                )
                
                # Generate execution summary
                summary_generator = SummaryGenerator()
                execution_data = {
                    'performance_metrics': performance_metrics,
                    'execution_time': execution_time,
                    'success': success,
                    'actions': extract_actions_from_context(context),
                    'results': extract_results_from_response(response),
                    'errors': [error_message] if error_message else []
                }
                
                summary_id = summary_generator.generate_summary(
                    agent_name, 
                    task_id, 
                    execution_data, 
                    log_content=response or '', 
                    user_query=user_query
                )
                
            except Exception as e:
                print(f"Agent management processing error: {e}", file=sys.stderr)
        
        # Cache execution results if available
        if CACHE_AVAILABLE:
            try:
                # Cache final thinking/results
                thinking_content = f"Post-execution results:\n"
                thinking_content += f"- Success: {success}\n"
                thinking_content += f"- Execution time: {execution_time:.2f}s\n"
                if response:
                    thinking_content += f"- Response preview: {response[:500]}...\n"
                if error_message:
                    thinking_content += f"- Error: {error_message}\n"
                
                cache_path = cache_thinking(
                    user_query,
                    thinking_content,
                    ["post-execute-hook", "completion", "results"]
                )
                if cache_path:
                    print(f"💾 Final results cached: {Path(cache_path).name}", file=sys.stderr)
                
                # Cache agent completion if agent was used
                if agent_name:
                    agent_cache_path = cache_agent(
                        agent_name,
                        {
                            "user_query": user_query,
                            "execution_result": response or "Execution completed",
                            "context": context or {},
                            "success": success,
                            "error_message": error_message
                        },
                        [
                            {"step": "post-execution", "timestamp": datetime.now().isoformat()},
                            {"step": "completion", "success": success, "execution_time": execution_time}
                        ],
                        {
                            "status": "completed" if success else "failed",
                            "result": response or "Execution completed",
                            "execution_time": execution_time,
                            "performance_metrics": locals().get('performance_metrics', {}),
                            "summary_id": summary_id
                        }
                    )
                    if agent_cache_path:
                        print(f"🤖 Agent completion cached: {Path(agent_cache_path).name}", file=sys.stderr)
                        
            except Exception as e:
                print(f"Warning: Failed to cache execution results: {e}", file=sys.stderr)
        
        # Clean up session file
        session_file.unlink(missing_ok=True)
        
        return {
            'log_filename': log_filename,
            'summary_id': summary_id,
            'agent_name': agent_name,
            'performance_metrics': performance_metrics if 'performance_metrics' in locals() else None
        }
        
    except Exception as e:
        print(f"Post-execute hook error: {e}", file=sys.stderr)
        return None

def calculate_performance_metrics(success: bool, execution_time: float, context: dict = None) -> dict:
    """Calculate performance metrics for the execution"""
    try:
        metrics = {}
        
        # Base success metric
        metrics['success_rate'] = 1.0 if success else 0.0
        
        # Time efficiency (inverse of execution time, normalized)
        # Shorter execution times get higher scores
        if execution_time > 0:
            metrics['time_efficiency'] = min(1.0, 60.0 / max(execution_time, 1.0))
        else:
            metrics['time_efficiency'] = 1.0
        
        # Quality indicators from context
        if context:
            # Tools usage efficiency
            tools_used = context.get('tools_used', [])
            if tools_used:
                avg_tool_duration = sum(tool.get('duration', 0) for tool in tools_used) / len(tools_used)
                metrics['tool_efficiency'] = min(1.0, 30.0 / max(avg_tool_duration, 1.0))
            else:
                metrics['tool_efficiency'] = 0.8  # Default for no tool usage
            
            # File access patterns
            files_accessed = context.get('files_accessed', [])
            if files_accessed:
                # More file accesses might indicate thoroughness, up to a point
                metrics['thoroughness'] = min(1.0, len(files_accessed) / 10.0)
            else:
                metrics['thoroughness'] = 0.5  # Default
        
        # Overall performance score
        metrics['overall_performance'] = sum(metrics.values()) / len(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Performance calculation error: {e}", file=sys.stderr)
        return {'success_rate': 1.0 if success else 0.0, 'overall_performance': 0.5}

def generate_completion_summary(success: bool, agent_name: str, metrics: dict, error_message: str = None) -> str:
    """Generate a completion summary for the task"""
    try:
        summary_parts = []
        
        if success:
            summary_parts.append(f"Successfully completed {agent_name} execution")
            
            # Add performance highlights
            if metrics:
                high_metrics = [k for k, v in metrics.items() if v > 0.8]
                if high_metrics:
                    summary_parts.append(f"High performance in: {', '.join(high_metrics)}")
        else:
            summary_parts.append(f"Failed {agent_name} execution")
            if error_message:
                summary_parts.append(f"Error: {error_message}")
        
        # Add performance summary
        if metrics and 'overall_performance' in metrics:
            perf_score = metrics['overall_performance']
            if perf_score > 0.9:
                summary_parts.append("Excellent performance achieved")
            elif perf_score > 0.7:
                summary_parts.append("Good performance achieved")
            else:
                summary_parts.append("Performance improvement opportunities identified")
        
        return '. '.join(summary_parts)
        
    except:
        return f"Execution completed for {agent_name}" if success else f"Execution failed for {agent_name}"

def extract_actions_from_context(context: dict = None) -> list:
    """Extract actions from execution context"""
    try:
        actions = []
        
        if context:
            # Extract from tools used
            tools_used = context.get('tools_used', [])
            for tool in tools_used:
                tool_name = tool.get('name', 'Unknown')
                actions.append(f"Used {tool_name} tool")
            
            # Extract from file operations
            files_accessed = context.get('files_accessed', [])
            for file_access in files_accessed:
                operation = file_access.get('operation', 'accessed')
                file_path = file_access.get('path', 'unknown file')
                actions.append(f"{operation.title()} {Path(file_path).name}")
            
            # Extract from agent invocations
            agents_used = context.get('agents_used', [])
            for agent in agents_used:
                agent_name = agent.get('name', 'Unknown')
                actions.append(f"Invoked {agent_name} agent")
        
        return actions[:10]  # Limit to 10 actions
        
    except:
        return ["Performed execution tasks"]

def extract_results_from_response(response: str = None) -> list:
    """Extract results from Claude's response"""
    try:
        results = []
        
        if response:
            # Look for result indicators in response
            result_patterns = [
                'successfully', 'completed', 'created', 'generated', 'found',
                'identified', 'analyzed', 'processed', 'updated'
            ]
            
            sentences = response.split('.')
            for sentence in sentences[:20]:  # Check first 20 sentences
                sentence = sentence.strip().lower()
                if any(pattern in sentence for pattern in result_patterns):
                    if len(sentence) > 20 and len(sentence) < 200:  # Reasonable length
                        results.append(sentence.capitalize())
        
        if not results:
            results = ["Task execution completed"]
            
        return results[:8]  # Limit to 8 results
        
    except:
        return ["Execution results generated"]

def extract_response_from_output():
    """Try to extract Claude's response from stdout/stderr"""
    try:
        # This is a simple implementation
        # In practice, this would need to be more sophisticated
        # to properly capture Claude's actual response
        return None
    except:
        return None

def determine_execution_success():
    """Determine if the execution was successful based on exit codes and environment"""
    try:
        # Check exit code
        exit_code = os.getenv('CLAUDE_EXIT_CODE', '0')
        if exit_code != '0':
            return False, f"Exit code: {exit_code}"
        
        # Check for error environment variables
        error_msg = os.getenv('CLAUDE_ERROR_MESSAGE')
        if error_msg:
            return False, error_msg
        
        return True, None
        
    except:
        return True, None

def collect_execution_context():
    """Collect execution context information"""
    context = {
        'end_time': datetime.now().isoformat(),
        'working_directory': os.getcwd(),
        'exit_code': os.getenv('CLAUDE_EXIT_CODE', '0'),
        'tools_used': [],
        'agents_used': [],
        'files_accessed': []
    }
    
    # Try to extract tool usage from environment
    tools_env = os.getenv('CLAUDE_TOOLS_USED')
    if tools_env:
        try:
            context['tools_used'] = json.loads(tools_env)
        except:
            pass
    
    # Try to extract agent usage from environment
    agents_env = os.getenv('CLAUDE_AGENTS_USED')
    if agents_env:
        try:
            context['agents_used'] = json.loads(agents_env)
        except:
            pass
    
    # Try to extract file access from environment
    files_env = os.getenv('CLAUDE_FILES_ACCESSED')
    if files_env:
        try:
            context['files_accessed'] = json.loads(files_env)
        except:
            pass
    
    return context

def main():
    """Main entry point for post-execute hook"""
    # Determine execution success
    success, error_message = determine_execution_success()
    
    # Extract Claude's response
    response = extract_response_from_output()
    
    # Collect context
    context = collect_execution_context()
    
    # Capture execution end
    result = capture_execution_end(success, error_message, response, context)
    
    # Provide feedback
    if result:
        if isinstance(result, dict):
            if result.get('log_filename'):
                print(f"📝 Execution logged to: {result['log_filename']}", file=sys.stderr)
            if result.get('summary_id'):
                print(f"📄 Summary generated: {result['summary_id']}", file=sys.stderr)
            if result.get('agent_name'):
                agent_name = result['agent_name']
                perf_metrics = result.get('performance_metrics', {})
                overall_perf = perf_metrics.get('overall_performance', 0) if perf_metrics else 0
                
                if overall_perf > 0.9:
                    print(f"🎯 {agent_name}: Excellent performance ({overall_perf:.1%})", file=sys.stderr)
                elif overall_perf > 0.7:
                    print(f"🎯 {agent_name}: Good performance ({overall_perf:.1%})", file=sys.stderr)
                elif overall_perf > 0:
                    print(f"🎯 {agent_name}: Performance tracked ({overall_perf:.1%})", file=sys.stderr)
        else:
            print(f"📝 Execution logged to: {result}", file=sys.stderr)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())