#!/usr/bin/env python3
"""
Claude Code Session Manager

Manages logging sessions across Claude Code executions.
Provides coordination between pre/post execute hooks and handles
session recovery, monitoring, and cleanup.

Author: Claude Code Research System
Version: 1.0.0
"""

import os
import sys
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import signal
import atexit

# Add scripts directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "scripts" / "logging"))

try:
    from claude_logger import get_claude_logger, end_logging_session
    LOGGING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Logging system not available: {e}", file=sys.stderr)
    LOGGING_AVAILABLE = False

class SessionManager:
    """Manages Claude Code logging sessions"""
    
    def __init__(self):
        self.session_dir = Path.home() / ".claude"
        self.session_file = self.session_dir / "current_session.json"
        self.active_sessions_file = self.session_dir / "active_sessions.json"
        
        # Ensure directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize active sessions tracking
        self.active_sessions = self._load_active_sessions()
        
        # Setup cleanup
        atexit.register(self.cleanup_on_exit)
    
    def _load_active_sessions(self) -> Dict[str, Any]:
        """Load active sessions from file"""
        try:
            if self.active_sessions_file.exists():
                with open(self.active_sessions_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load active sessions: {e}", file=sys.stderr)
        
        return {}
    
    def _save_active_sessions(self):
        """Save active sessions to file"""
        try:
            with open(self.active_sessions_file, 'w') as f:
                json.dump(self.active_sessions, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save active sessions: {e}", file=sys.stderr)
    
    def register_session(self, session_data: Dict[str, Any]) -> bool:
        """Register a new active session"""
        try:
            session_id = session_data.get('session_id')
            if not session_id:
                return False
            
            # Add to active sessions
            self.active_sessions[session_id] = {
                **session_data,
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Save to file
            self._save_active_sessions()
            
            return True
            
        except Exception as e:
            print(f"Error registering session: {e}", file=sys.stderr)
            return False
    
    def unregister_session(self, session_id: str) -> bool:
        """Unregister an active session"""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'completed'
                self.active_sessions[session_id]['completed_at'] = datetime.now().isoformat()
                
                # Keep for a short time then remove
                del self.active_sessions[session_id]
                self._save_active_sessions()
                
                return True
        except Exception as e:
            print(f"Error unregistering session: {e}", file=sys.stderr)
        
        return False
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get current session data"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error reading current session: {e}", file=sys.stderr)
        
        return None
    
    def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """Clean up stale sessions that never completed"""
        if not LOGGING_AVAILABLE:
            return
        
        try:
            current_time = datetime.now()
            stale_sessions = []
            
            for session_id, session_data in self.active_sessions.items():
                registered_at = datetime.fromisoformat(session_data.get('registered_at', ''))
                age = current_time - registered_at
                
                if age > timedelta(hours=max_age_hours):
                    stale_sessions.append(session_id)
                    
                    # Check if process is still running
                    pid = session_data.get('pid')
                    if pid and not self._is_process_running(pid):
                        # Process is dead, force end the logging session
                        try:
                            end_logging_session(False, f"Process {pid} terminated unexpectedly")
                            print(f"Cleaned up stale session: {session_id}", file=sys.stderr)
                        except:
                            pass
            
            # Remove stale sessions
            for session_id in stale_sessions:
                self.active_sessions.pop(session_id, None)
            
            if stale_sessions:
                self._save_active_sessions()
                
        except Exception as e:
            print(f"Error cleaning up stale sessions: {e}", file=sys.stderr)
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running"""
        try:
            return psutil.pid_exists(pid)
        except:
            return False
    
    def monitor_sessions(self, check_interval: int = 300):  # 5 minutes
        """Monitor active sessions and clean up stale ones"""
        def monitor_loop():
            while True:
                try:
                    self.cleanup_stale_sessions()
                    time.sleep(check_interval)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Session monitor error: {e}", file=sys.stderr)
                    time.sleep(check_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about active sessions"""
        stats = {
            'active_sessions_count': len([s for s in self.active_sessions.values() if s.get('status') == 'active']),
            'total_tracked_sessions': len(self.active_sessions),
            'oldest_active_session': None,
            'newest_active_session': None
        }
        
        active_sessions = [s for s in self.active_sessions.values() if s.get('status') == 'active']
        
        if active_sessions:
            # Find oldest and newest
            sorted_by_time = sorted(active_sessions, key=lambda x: x.get('registered_at', ''))
            stats['oldest_active_session'] = sorted_by_time[0]
            stats['newest_active_session'] = sorted_by_time[-1]
        
        return stats
    
    def force_end_session(self, session_id: str, reason: str = "Forced termination") -> bool:
        """Force end a session"""
        if not LOGGING_AVAILABLE:
            return False
        
        try:
            # End the logging session
            end_logging_session(False, reason)
            
            # Unregister from active sessions
            self.unregister_session(session_id)
            
            # Clean up current session file if it matches
            current_session = self.get_current_session()
            if current_session and current_session.get('session_id') == session_id:
                self.session_file.unlink(missing_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Error force ending session {session_id}: {e}", file=sys.stderr)
            return False
    
    def cleanup_on_exit(self):
        """Cleanup function called on exit"""
        try:
            # End current session if exists
            current_session = self.get_current_session()
            if current_session:
                session_id = current_session.get('session_id')
                if session_id:
                    self.force_end_session(session_id, "Process terminated")
            
        except Exception as e:
            print(f"Cleanup error: {e}", file=sys.stderr)
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [s for s in self.active_sessions.values() if s.get('status') == 'active']
    
    def export_session_data(self, output_path: str) -> bool:
        """Export session data for debugging"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'active_sessions': self.active_sessions,
                'current_session': self.get_current_session(),
                'stats': self.get_session_stats()
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Export error: {e}", file=sys.stderr)
            return False

def main():
    """Command line interface for session manager"""
    parser = argparse.ArgumentParser(description="Claude Code Session Manager")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show session status')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List active sessions')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up stale sessions')
    cleanup_parser.add_argument('--max-age-hours', type=int, default=24,
                               help='Maximum age for sessions before cleanup')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start session monitoring')
    monitor_parser.add_argument('--interval', type=int, default=300,
                               help='Check interval in seconds')
    
    # Kill command
    kill_parser = subparsers.add_parser('kill', help='Force end a session')
    kill_parser.add_argument('session_id', help='Session ID to kill')
    kill_parser.add_argument('--reason', default='Manual termination',
                            help='Reason for termination')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export session data')
    export_parser.add_argument('output_path', help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    try:
        manager = SessionManager()
        
        if args.command == 'status':
            stats = manager.get_session_stats()
            current_session = manager.get_current_session()
            
            print("📊 Session Manager Status")
            print("=" * 40)
            print(f"Active Sessions: {stats['active_sessions_count']}")
            print(f"Total Tracked: {stats['total_tracked_sessions']}")
            
            if current_session:
                print(f"\n🔄 Current Session:")
                print(f"  ID: {current_session.get('session_id', 'Unknown')}")
                print(f"  Query: {current_session.get('user_query', 'Unknown')}")
                print(f"  Started: {current_session.get('start_time', 'Unknown')}")
            else:
                print("\n✅ No current session")
            
            if stats['oldest_active_session']:
                oldest = stats['oldest_active_session']
                print(f"\n⏰ Oldest Active Session:")
                print(f"  Started: {oldest.get('registered_at', 'Unknown')}")
                print(f"  Query: {oldest.get('user_query', 'Unknown')}")
        
        elif args.command == 'list':
            active_sessions = manager.list_active_sessions()
            
            if not active_sessions:
                print("No active sessions found.")
            else:
                print(f"📋 Active Sessions ({len(active_sessions)}):")
                print("-" * 60)
                
                for i, session in enumerate(active_sessions, 1):
                    registered_at = session.get('registered_at', 'Unknown')
                    try:
                        dt = datetime.fromisoformat(registered_at)
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        formatted_time = registered_at
                    
                    print(f"{i}. {session.get('session_id', 'Unknown')}")
                    print(f"   Started: {formatted_time}")
                    print(f"   Query: {session.get('user_query', 'Unknown')}")
                    print(f"   PID: {session.get('pid', 'Unknown')}")
                    print()
        
        elif args.command == 'cleanup':
            print(f"🧹 Cleaning up sessions older than {args.max_age_hours} hours...")
            manager.cleanup_stale_sessions(args.max_age_hours)
            print("✅ Cleanup completed")
        
        elif args.command == 'monitor':
            print(f"👁️  Starting session monitoring (interval: {args.interval}s)")
            print("Press Ctrl+C to stop...")
            
            monitor_thread = manager.monitor_sessions(args.interval)
            
            try:
                while monitor_thread.is_alive():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping monitor...")
        
        elif args.command == 'kill':
            print(f"💀 Force ending session: {args.session_id}")
            if manager.force_end_session(args.session_id, args.reason):
                print("✅ Session terminated")
            else:
                print("❌ Failed to terminate session")
        
        elif args.command == 'export':
            print(f"📤 Exporting session data to: {args.output_path}")
            if manager.export_session_data(args.output_path):
                print("✅ Export completed")
            else:
                print("❌ Export failed")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())