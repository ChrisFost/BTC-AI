"""
env_observation.py - Part 1: Environment Observation Core and Setup

This module provides tools for monitoring and observing the simulation environment,
tracking agent behaviors, and collecting metrics on environment states.
"""

import logging
import time
import threading
import json
from collections import deque, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set, Callable, TypeVar, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
import importlib
import pandas as pd

# Use TYPE_CHECKING for imports used only for type checking
if TYPE_CHECKING:
    from src.environment.env_risk import RiskManager, RiskLevel, RiskEvent
    # This import is only used for type hints and not at runtime
    from typing import TypeVar
    Agent = TypeVar('Agent')
else:
    # For runtime, load risk manager dynamically when needed
    try:
        env_risk_module = importlib.import_module("src.environment.env_risk")
        RiskManager = env_risk_module.RiskManager
        RiskLevel = env_risk_module.RiskLevel
        RiskEvent = env_risk_module.RiskEvent
    except ImportError as e:
        # Define placeholder classes if the risk module cannot be imported
        class RiskManager: pass
        class RiskLevel: pass
        class RiskEvent: pass

class EnvObservation:
    """Base class for environment observations."""
    def __init__(self, env_id: str = ""):
        self.env_id = env_id
        self.timestamp = time.time()
        self.data = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary format."""
        return {
            "env_id": self.env_id,
            "timestamp": self.timestamp,
            "data": self.data
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvObservation':
        """Create observation from dictionary format."""
        obs = cls(data.get("env_id", ""))
        obs.timestamp = data.get("timestamp", time.time())
        obs.data = data.get("data", {})
        return obs

# Local imports - removed imports for classes that don't exist in env_base.py
# Instead defining these classes directly here
@dataclass
class Environment:
    """Base environment interface"""
    env_id: str = ""
    
@dataclass
class EnvironmentState:
    """Environment state container"""
    step: int = 0
    
@dataclass
class ObservationSpace:
    """Observation space descriptor"""
    shape: Tuple = field(default_factory=tuple)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("environment_observation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("env_observation")

@dataclass
class ObservationPoint:
    """A single observation data point from the environment."""
    timestamp: float
    environment_id: str
    observation_type: str
    data: Dict[str, Any]
    agent_id: Optional[str] = None
    sequence_id: Optional[int] = None
    tags: Set[str] = field(default_factory=set)

@dataclass
class ObservationConfig:
    """Configuration for the observation system."""
    sampling_rate: float = 1.0  # Samples per second
    buffer_size: int = 1000     # Size of observation buffer
    detailed_logging: bool = False
    batch_size: int = 50        # Number of observations to process in batch
    event_triggers: Dict[str, Callable] = field(default_factory=dict)
    critical_events: Set[str] = field(default_factory=set)
    persistence_path: Optional[str] = "observations/"

class ObservationSystem:
    """
    Main system for monitoring and observing the environment.
    Collects data points, processes metrics, and provides analysis tools.
    """
    
    def __init__(self, config=None):
        """Initialize the observation system with configuration parameters."""
        self.observations = deque(maxlen=1000)  # Default buffer size
        self.running = False
        self.observation_thread = None
        self.sequence_counter = 0
        self.registered_environments = {}
        self.registered_callbacks = {}
        self.last_observation_time = 0
        
        # Agent tracking
        self.tracked_agents: Dict[str, Any] = {}
        
    def register_environment(self, env: Environment) -> None:
        """Register an environment to be observed."""
        env_id = env.env_id
        self.registered_environments[env_id] = env
        self.observation_spaces[env_id] = env.shape
        logger.info(f"Registered environment {env_id} for observation")
        
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback function for a specific event type."""
        if event_type not in self.registered_callbacks:
            self.registered_callbacks[event_type] = []
        self.registered_callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for event: {event_type}")
        
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """Unregister a callback function."""
        if event_type in self.registered_callbacks:
            if callback in self.registered_callbacks[event_type]:
                self.registered_callbacks[event_type].remove(callback)
                logger.debug(f"Unregistered callback for event: {event_type}")
    
    def record_observation(self, env_id: str, obs_type: str, 
                          data: Dict[str, Any], agent_id: str = None, 
                          tags: Set[str] = None) -> ObservationPoint:
        """Record a single observation from the environment."""
        self.sequence_counter += 1
        observation = ObservationPoint(
            timestamp=time.time(),
            environment_id=env_id,
            observation_type=obs_type,
            data=data,
            agent_id=agent_id,
            sequence_id=self.sequence_counter,
            tags=tags or set()
        )
        
        self.observations.append(observation)
        self.last_observation_time = observation.timestamp
        
        # Process callbacks for this observation type
        if obs_type in self.registered_callbacks:
            for callback in self.registered_callbacks[obs_type]:
                try:
                    callback(observation)
                except Exception as e:
                    logger.error(f"Error in callback for {obs_type}: {e}")
        
        # Check for critical events that need immediate attention
        if obs_type in self.config.critical_events:
            logger.warning(f"Critical event observed: {obs_type} in env {env_id}")
            
        return observation
    
    def start_monitoring(self) -> None:
        """Start the continuous monitoring thread."""
        if self.running:
            logger.warning("Monitoring already running")
            return
            
        self.running = True
        self.observation_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.observation_thread.start()
        logger.info("Started continuous environment monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the continuous monitoring thread."""
        self.running = False
        if self.observation_thread:
            self.observation_thread.join(timeout=5.0)
            logger.info("Stopped continuous environment monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that runs in a separate thread."""
        sample_interval = 1.0 / self.config.sampling_rate
        
        while self.running:
            start_time = time.time()
            
            try:
                # Poll all registered environments
                for env_id, env in self.registered_environments.items():
                    # Get current environment state
                    state = env.get_state()
                    
                    # Record basic environment state
                    self.record_observation(
                        env_id=env_id,
                        obs_type="environment_state",
                        data={
                            "state": state.to_dict() if hasattr(state, "to_dict") else str(state),
                            "timestamp": time.time(),
                            "active_agents": env.get_active_agents() if hasattr(env, "get_active_agents") else [],
                        },
                        tags={"routine", "state_snapshot"}
                    )
                    
                    # Check for any environment-specific observations
                    if hasattr(env, "get_observations"):
                        custom_observations = env.get_observations()
                        for obs_type, obs_data in custom_observations.items():
                            self.record_observation(
                                env_id=env_id,
                                obs_type=obs_type,
                                data=obs_data,
                                tags={"custom", obs_type}
                            )
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for the remaining interval
            elapsed = time.time() - start_time
            if elapsed < sample_interval:
                time.sleep(sample_interval - elapsed)
    
    def get_recent_observations(self, count: int = 10, 
                               obs_type: str = None,
                               env_id: str = None,
                               agent_id: str = None) -> List[ObservationPoint]:
        """Get the most recent observations with optional filtering."""
        filtered = self.observations
        
        if obs_type:
            filtered = [o for o in filtered if o.observation_type == obs_type]
        if env_id:
            filtered = [o for o in filtered if o.environment_id == env_id]
        if agent_id:
            filtered = [o for o in filtered if o.agent_id == agent_id]
            
        # Return the most recent ones up to count
        return list(filtered)[-count:]
    
    def persist_observations(self, filename: str = None) -> str:
        """Save current observations to disk."""
        if not filename:
            timestamp = int(time.time())
            filename = f"{self.config.persistence_path}observations_{timestamp}.json"
            
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert observations to serializable format
        serializable = []
        for obs in self.observations:
            serializable.append({
                "timestamp": obs.timestamp,
                "environment_id": obs.environment_id,
                "observation_type": obs.observation_type,
                "data": obs.data,
                "agent_id": obs.agent_id,
                "sequence_id": obs.sequence_id,
                "tags": list(obs.tags)
            })
            
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
            
        logger.info(f"Persisted {len(serializable)} observations to {filename}")
        return filename
    
    def load_observations(self, filename: str) -> None:
        """Load observations from disk."""
        with open(filename, 'r') as f:
            data = json.load(f)
            
        # Convert back to ObservationPoint objects
        for item in data:
            obs = ObservationPoint(
                timestamp=item["timestamp"],
                environment_id=item["environment_id"],
                observation_type=item["observation_type"],
                data=item["data"],
                agent_id=item["agent_id"],
                sequence_id=item["sequence_id"],
                tags=set(item["tags"])
            )
            self.observations.append(obs)
            
        # Update sequence counter
        if self.observations:
            max_seq = max(obs.sequence_id for obs in self.observations if obs.sequence_id is not None)
            self.sequence_counter = max(self.sequence_counter, max_seq)
            
        logger.info(f"Loaded {len(data)} observations from {filename}")

# Helper functions
def create_default_observation_system() -> ObservationSystem:
    """Create an observation system with default configuration."""
    config = ObservationConfig(
        sampling_rate=2.0,  # 2 samples per second
        buffer_size=10000,  # Store last 10000 observations
        critical_events={"risk_warning", "agent_error", "environment_error"}
    )
    return ObservationSystem(config)


class AgentTracker:
    """Tracks and records agent states and actions over time."""
    
    def __init__(self, observation_system):
        """Initialize the agent tracker with a reference to the observation system."""
        self.observation_system = observation_system
        self.agent_history: Dict[str, List[Dict[str, Any]]] = {}
        self.tracked_agents: Dict[str, Any] = {}
        self.action_counts: Dict[str, Dict[str, int]] = {}
        self.state_timestamps: Dict[str, Dict[str, float]] = {}
        
    def register_agent(self, agent: Any) -> None:
        """Register an agent to be tracked."""
        agent_id = getattr(agent, 'id', str(id(agent)))
        self.tracked_agents[agent_id] = agent
        self.agent_history[agent_id] = []
        self.action_counts[agent_id] = {}
        self.state_timestamps[agent_id] = {}
        
        # Register callback for agent state changes
        def on_agent_state_change(observation):
            if observation.agent_id == agent_id:
                self._process_agent_state_change(agent_id, observation.data)
                
        self.observation_system.register_callback(
            "agent_state_change", on_agent_state_change
        )
        
        logger.info(f"Started tracking agent {agent_id}")
        
    def _process_agent_state_change(self, agent_id: str, state_data: Dict[str, Any]) -> None:
        """Process an agent state change observation."""
        if agent_id not in self.agent_history:
            return
            
        # Record the state in history
        self.agent_history[agent_id].append({
            "timestamp": time.time(),
            "state": state_data
        })
        
        # Update state timestamps
        if "state" in state_data:
            state = state_data["state"]
            self.state_timestamps[agent_id][state] = time.time()
            
        # Limit history size
        max_history = 1000  # Could be configurable
        if len(self.agent_history[agent_id]) > max_history:
            self.agent_history[agent_id] = self.agent_history[agent_id][-max_history:]
    
    def record_agent_action(self, agent_id: str, action: str, 
                           result: Any = None, context: Dict[str, Any] = None) -> None:
        """Record an action taken by an agent."""
        if agent_id not in self.tracked_agents:
            logger.warning(f"Attempting to record action for untracked agent {agent_id}")
            return
            
        # Update action counts
        if action not in self.action_counts[agent_id]:
            self.action_counts[agent_id][action] = 0
        self.action_counts[agent_id][action] += 1
        
        # Record observation
        self.observation_system.record_observation(
            env_id=self.tracked_agents[agent_id].environment_id,
            obs_type="agent_action",
            data={
                "action": action,
                "result": result,
                "context": context or {}
            },
            agent_id=agent_id,
            tags={"agent", "action", action}
        )
    
    def get_agent_history(self, agent_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get the history of an agent's states."""
        if agent_id not in self.agent_history:
            return []
            
        history = self.agent_history[agent_id]
        if limit:
            history = history[-limit:]
        return history
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics about an agent's behavior."""
        if agent_id not in self.tracked_agents:
            return {}
            
        # Calculate time in each state
        current_time = time.time()
        state_durations = {}
        for state, timestamp in self.state_timestamps[agent_id].items():
            state_durations[state] = current_time - timestamp
            
        return {
            "action_counts": self.action_counts[agent_id],
            "state_durations": state_durations,
            "total_actions": sum(self.action_counts[agent_id].values()),
            "history_length": len(self.agent_history[agent_id]),
            "most_common_action": max(
                self.action_counts[agent_id].items(), 
                key=lambda x: x[1]
            )[0] if self.action_counts[agent_id] else None
        }
    
    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked agents."""
        return {
            agent_id: self.get_agent_stats(agent_id)
            for agent_id in self.tracked_agents
        }
        
    def export_agent_data(self, agent_id: str, filename: str = None) -> str:
        """Export agent history to a file."""
        if agent_id not in self.agent_history:
            raise ValueError(f"Agent {agent_id} not found in tracker")
            
        data = {
            "agent_id": agent_id,
            "history": self.agent_history[agent_id],
            "action_counts": self.action_counts[agent_id],
            "stats": self.get_agent_stats(agent_id)
        }
        
        if not filename:
            timestamp = int(time.time())
            filename = f"agent_{agent_id}_{timestamp}.json"
            
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        return filename


class EnvironmentStateSnapshot:
    """Captures and stores point-in-time snapshots of environment state."""
    
    def __init__(self, env: Environment, timestamp: float = None):
        """Create a new snapshot of the given environment."""
        self.environment_id = env.env_id
        self.timestamp = timestamp or time.time()
        self.state = env.get_state()
        self.active_agents = env.get_active_agents() if hasattr(env, "get_active_agents") else []
        self.snapshot_id = f"{self.environment_id}_{int(self.timestamp)}"
        
        # Capture environment-specific metrics if available
        self.metrics = {}
        if hasattr(env, "get_metrics"):
            self.metrics = env.get_metrics()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to a dictionary for serialization."""
        return {
            "environment_id": self.environment_id,
            "timestamp": self.timestamp,
            "state": self.state.to_dict() if hasattr(self.state, "to_dict") else str(self.state),
            "active_agents": self.active_agents,
            "snapshot_id": self.snapshot_id,
            "metrics": self.metrics
        }


class EnvironmentTracker:
    """Tracks environment states and transitions over time."""
    
    def __init__(self, observation_system):
        """Initialize the environment tracker."""
        self.observation_system = observation_system
        self.snapshots: Dict[str, List[EnvironmentStateSnapshot]] = {}
        self.snapshot_interval = 60.0  # Seconds between snapshots
        self.last_snapshot_time: Dict[str, float] = {}
        self.is_recording = False
        self.snapshot_thread = None
        
    def register_environment(self, env: Environment) -> None:
        """Register an environment to be tracked."""
        env_id = env.env_id
        self.snapshots[env_id] = []
        self.last_snapshot_time[env_id] = 0
        logger.info(f"Environment tracker registered environment {env_id}")
        
    def take_snapshot(self, env_id: str) -> Optional[EnvironmentStateSnapshot]:
        """Take a snapshot of the current environment state."""
        if env_id not in self.observation_system.registered_environments:
            logger.warning(f"Attempting to snapshot unregistered environment {env_id}")
            return None
            
        env = self.observation_system.registered_environments[env_id]
        snapshot = EnvironmentStateSnapshot(env)
        self.snapshots[env_id].append(snapshot)
        self.last_snapshot_time[env_id] = snapshot.timestamp
        
        # Record observation for the snapshot
        self.observation_system.record_observation(
            env_id=env_id,
            obs_type="environment_snapshot",
            data=snapshot.to_dict(),
            tags={"environment", "snapshot", "state"}
        )
        
        # Limit snapshots kept in memory
        max_snapshots = 100  # Could be configurable
        if len(self.snapshots[env_id]) > max_snapshots:
            self.snapshots[env_id] = self.snapshots[env_id][-max_snapshots:]
            
        return snapshot
    
    def start_recording(self) -> None:
        """Start automatic periodic snapshots of all environments."""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.snapshot_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True
        )
        self.snapshot_thread.start()
        logger.info("Started automatic environment state recording")
        
    def stop_recording(self) -> None:
        """Stop automatic periodic snapshots."""
        self.is_recording = False
        if self.snapshot_thread:
            self.snapshot_thread.join(timeout=5.0)
            logger.info("Stopped automatic environment recording")
            
    def _recording_loop(self) -> None:
        """Main loop for periodic environment snapshots."""
        while self.is_recording:
            current_time = time.time()
            
            for env_id in self.observation_system.registered_environments:
                # Check if enough time has passed since last snapshot
                if current_time - self.last_snapshot_time.get(env_id, 0) >= self.snapshot_interval:
                    try:
                        self.take_snapshot(env_id)
                    except Exception as e:
                        logger.error(f"Error taking snapshot of environment {env_id}: {e}")
            
            # Sleep a bit to avoid CPU spinning
            time.sleep(1.0)
            
    def get_snapshots(self, env_id: str, start_time: float = None, 
                     end_time: float = None, limit: int = None) -> List[EnvironmentStateSnapshot]:
        """Get snapshots for an environment, optionally filtered by time range."""
        if env_id not in self.snapshots:
            return []
            
        filtered = self.snapshots[env_id]
        
        if start_time:
            filtered = [s for s in filtered if s.timestamp >= start_time]
        if end_time:
            filtered = [s for s in filtered if s.timestamp <= end_time]
            
        # Sort by timestamp (newest first)
        filtered = sorted(filtered, key=lambda s: s.timestamp, reverse=True)
        
        if limit:
            filtered = filtered[:limit]
            
        return filtered
        
    def compare_snapshots(self, snapshot1: EnvironmentStateSnapshot, 
                         snapshot2: EnvironmentStateSnapshot) -> Dict[str, Any]:
        """Compare two snapshots and return the differences."""
        if snapshot1.environment_id != snapshot2.environment_id:
            raise ValueError("Cannot compare snapshots from different environments")
            
        # This is a simple implementation - could be made more sophisticated
        # to detect specific types of changes based on environment type
        return {
            "environment_id": snapshot1.environment_id,
            "time_difference": snapshot2.timestamp - snapshot1.timestamp,
            "agent_changes": {
                "added": [a for a in snapshot2.active_agents if a not in snapshot1.active_agents],
                "removed": [a for a in snapshot1.active_agents if a not in snapshot2.active_agents]
            },
            "metric_changes": {
                metric: {
                    "before": snapshot1.metrics.get(metric),
                    "after": snapshot2.metrics.get(metric),
                    "change": snapshot2.metrics.get(metric) - snapshot1.metrics.get(metric)
                    if metric in snapshot1.metrics and metric in snapshot2.metrics
                    and isinstance(snapshot1.metrics.get(metric), (int, float))
                    and isinstance(snapshot2.metrics.get(metric), (int, float))
                    else "N/A"
                }
                for metric in set(list(snapshot1.metrics.keys()) + list(snapshot2.metrics.keys()))
            }
        }


class InteractionTracker:
    """Tracks interactions between agents and environment elements."""
    
    def __init__(self, observation_system):
        """Initialize the interaction tracker."""
        self.observation_system = observation_system
        self.interactions = []
        self.interaction_counts = {}
        
    def record_interaction(self, source_id: str, target_id: str, 
                          interaction_type: str, data: Dict[str, Any] = None,
                          env_id: str = None) -> None:
        """Record an interaction between two entities in the environment."""
        timestamp = time.time()
        interaction_id = f"{source_id}_{target_id}_{interaction_type}_{int(timestamp)}"
        
        interaction = {
            "interaction_id": interaction_id,
            "source_id": source_id,
            "target_id": target_id,
            "type": interaction_type,
            "timestamp": timestamp,
            "data": data or {}
        }
        
        self.interactions.append(interaction)
        
        # Update interaction counts
        key = (source_id, target_id, interaction_type)
        if key not in self.interaction_counts:
            self.interaction_counts[key] = 0
        self.interaction_counts[key] += 1
        
        # Record as observation
        self.observation_system.record_observation(
            env_id=env_id or "unknown",
            obs_type="interaction",
            data=interaction,
            tags={"interaction", interaction_type}
        )
        
    def get_interactions(self, source_id: str = None, target_id: str = None,
                       interaction_type: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Get interactions, optionally filtered by criteria."""
        filtered = self.interactions
        
        if source_id:
            filtered = [i for i in filtered if i["source_id"] == source_id]
        if target_id:
            filtered = [i for i in filtered if i["target_id"] == target_id]
        if interaction_type:
            filtered = [i for i in filtered if i["type"] == interaction_type]
            
        # Sort by timestamp (newest first)
        filtered = sorted(filtered, key=lambda i: i["timestamp"], reverse=True)
        
        if limit:
            filtered = filtered[:limit]
            
        return filtered
        
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked interactions."""
        if not self.interactions:
            return {
                "total_interactions": 0,
                "unique_sources": 0,
                "unique_targets": 0,
                "interaction_types": []
            }
            
        sources = set(i["source_id"] for i in self.interactions)
        targets = set(i["target_id"] for i in self.interactions)
        types = set(i["type"] for i in self.interactions)
        
        # Count by type
        type_counts = {}
        for i_type in types:
            type_counts[i_type] = len([i for i in self.interactions if i["type"] == i_type])
            
        # Most active source and target
        source_counts = {}
        for source in sources:
            source_counts[source] = len([i for i in self.interactions if i["source_id"] == source])
            
        target_counts = {}
        for target in targets:
            target_counts[target] = len([i for i in self.interactions if i["target_id"] == target])
            
        return {
            "total_interactions": len(self.interactions),
            "unique_sources": len(sources),
            "unique_targets": len(targets),
            "interaction_types": list(types),
            "type_counts": type_counts,
            "most_active_source": max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None,
            "most_active_target": max(target_counts.items(), key=lambda x: x[1])[0] if target_counts else None
        }


class MetricsCollector:
    """Collects and manages metrics from the environment and agents."""
    
    def __init__(self, observation_system):
        """Initialize the metrics collector."""
        self.observation_system = observation_system
        self.metrics: Dict[str, Dict[str, List[Tuple[float, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.metric_configs: Dict[str, Dict[str, Any]] = {}
        self.derived_metrics: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.aggregation_intervals = [60, 300, 900, 3600]  # 1min, 5min, 15min, 1hr
        self.aggregated_metrics: Dict[int, Dict[str, Dict[str, List[Tuple[float, Any]]]]] = {
            interval: defaultdict(lambda: defaultdict(list)) for interval in self.aggregation_intervals
        }
        self.last_aggregation_time = {interval: 0 for interval in self.aggregation_intervals}
        self.running = False
        self.aggregation_thread = None
        
        # Register for environment observations
        self.observation_system.register_callback(
            "environment_state", self._on_environment_observation
        )
        self.observation_system.register_callback(
            "agent_state_change", self._on_agent_observation
        )
        
    def register_metric(self, name: str, scope: str, 
                       config: Dict[str, Any] = None) -> None:
        """
        Register a new metric to be tracked.
        
        Args:
            name: The name of the metric
            scope: The scope of the metric (e.g., "environment", "agent", "global")
            config: Optional configuration for the metric
        """
        if scope not in self.metrics:
            self.metrics[scope] = defaultdict(list)
            
        self.metric_configs[(scope, name)] = config or {}
        logger.info(f"Registered metric {name} for scope {scope}")
        
    def register_derived_metric(self, name: str, scope: str, 
                               calculation_func: Callable, 
                               source_metrics: List[Tuple[str, str]]) -> None:
        """
        Register a derived metric that is calculated from other metrics.
        
        Args:
            name: The name of the derived metric
            scope: The scope of the metric
            calculation_func: A function that calculates the derived metric
            source_metrics: List of (scope, name) tuples for the source metrics
        """
        self.derived_metrics[scope][name] = {
            "calculation_func": calculation_func,
            "source_metrics": source_metrics
        }
        logger.info(f"Registered derived metric {name} for scope {scope}")
        
    def record_metric(self, name: str, scope: str, 
                     value: Any, timestamp: float = None) -> None:
        """
        Record a value for a metric.
        
        Args:
            name: The name of the metric
            scope: The scope of the metric
            value: The metric value
            timestamp: Optional timestamp, defaults to current time
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.metrics[scope][name].append((timestamp, value))
        
        # Apply any configured limits to the number of data points kept
        config = self.metric_configs.get((scope, name), {})
        max_points = config.get("max_points", 10000)
        if len(self.metrics[scope][name]) > max_points:
            # Keep only the most recent points
            self.metrics[scope][name] = self.metrics[scope][name][-max_points:]
            
        # Calculate derived metrics that depend on this one
        self._update_derived_metrics(scope, name)
        
    def _update_derived_metrics(self, source_scope: str, source_name: str) -> None:
        """Update any derived metrics that depend on the given source metric."""
        for scope, metrics in self.derived_metrics.items():
            for name, config in metrics.items():
                if (source_scope, source_name) in config["source_metrics"]:
                    try:
                        # Gather the latest values of all source metrics
                        source_values = {}
                        for src_scope, src_name in config["source_metrics"]:
                            if (src_scope in self.metrics and 
                                src_name in self.metrics[src_scope] and 
                                self.metrics[src_scope][src_name]):
                                # Get the most recent value
                                _, value = self.metrics[src_scope][src_name][-1]
                                source_values[(src_scope, src_name)] = value
                                
                        # Skip if not all source metrics are available
                        if len(source_values) != len(config["source_metrics"]):
                            continue
                            
                        # Calculate and record the derived metric
                        derived_value = config["calculation_func"](source_values)
                        self.record_metric(name, scope, derived_value)
                    except Exception as e:
                        logger.error(f"Error calculating derived metric {scope}.{name}: {e}")
    
    def _on_environment_observation(self, observation) -> None:
        """Process environment observations to extract metrics."""
        env_id = observation.environment_id
        data = observation.data
        
        # Extract state-based metrics
        if "state" in data and isinstance(data["state"], dict):
            for key, value in data["state"].items():
                if isinstance(value, (int, float)):
                    # Only record numeric values automatically
                    self.record_metric(key, f"environment.{env_id}", value, observation.timestamp)
        
        # Extract any explicit metrics
        if "metrics" in data and isinstance(data["metrics"], dict):
            for key, value in data["metrics"].items():
                self.record_metric(key, f"environment.{env_id}", value, observation.timestamp)
    
    def _on_agent_observation(self, observation) -> None:
        """Process agent observations to extract metrics."""
        agent_id = observation.agent_id
        if not agent_id:
            return
            
        data = observation.data
        
        # Extract agent state metrics
        if "state" in data:
            self.record_metric("state", f"agent.{agent_id}", data["state"], observation.timestamp)
            
        # Extract any explicit metrics
        if "metrics" in data and isinstance(data["metrics"], dict):
            for key, value in data["metrics"].items():
                self.record_metric(key, f"agent.{agent_id}", value, observation.timestamp)
    
    def start_aggregation(self) -> None:
        """Start the metric aggregation thread."""
        if self.running:
            return
            
        self.running = True
        self.aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        self.aggregation_thread.start()
        logger.info("Started metric aggregation")
        
    def stop_aggregation(self) -> None:
        """Stop the metric aggregation thread."""
        self.running = False
        if self.aggregation_thread:
            self.aggregation_thread.join(timeout=5.0)
            logger.info("Stopped metric aggregation")
    
    def _aggregation_loop(self) -> None:
        """Main loop for periodic metric aggregation."""
        while self.running:
            current_time = time.time()
            
            # Check each aggregation interval
            for interval in self.aggregation_intervals:
                # Only aggregate if the interval has elapsed
                if current_time - self.last_aggregation_time[interval] >= interval:
                    self._perform_aggregation(interval, current_time)
                    self.last_aggregation_time[interval] = current_time
            
            # Sleep briefly to avoid CPU spinning
            time.sleep(1.0)
            
    def _perform_aggregation(self, interval: int, current_time: float) -> None:
        """Perform metric aggregation for the given interval."""
        # Calculate the start time for this aggregation period
        start_time = current_time - interval
        
        # Process each metric
        for scope in self.metrics:
            for name in self.metrics[scope]:
                # Get data points within the interval
                points = [(t, v) for t, v in self.metrics[scope][name] if t >= start_time]
                
                if not points:
                    continue
                    
                # Extract values for easier calculation
                values = [v for _, v in points if isinstance(v, (int, float))]
                
                if not values:
                    continue
                    
                # Calculate aggregations
                aggregations = {
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values),
                    "sum": np.sum(values),
                    "std": np.std(values) if len(values) > 1 else 0
                }
                
                # Store each aggregation as a separate metric
                for agg_name, agg_value in aggregations.items():
                    metric_name = f"{name}_{agg_name}_{interval}s"
                    self.aggregated_metrics[interval][scope][metric_name].append(
                        (current_time, agg_value)
                    )
                    
                    # Limit the number of aggregated points kept
                    max_agg_points = 1000  # Could be configurable
                    if len(self.aggregated_metrics[interval][scope][metric_name]) > max_agg_points:
                        self.aggregated_metrics[interval][scope][metric_name] = \
                            self.aggregated_metrics[interval][scope][metric_name][-max_agg_points:]
    
    def get_metric_values(self, name: str, scope: str, 
                         start_time: float = None, end_time: float = None) -> List[Tuple[float, Any]]:
        """Get values for a specific metric within an optional time range."""
        if scope not in self.metrics or name not in self.metrics[scope]:
            return []
            
        values = self.metrics[scope][name]
        
        if start_time:
            values = [v for v in values if v[0] >= start_time]
        if end_time:
            values = [v for v in values if v[0] <= end_time]
            
        return values
    
    def get_aggregated_values(self, name: str, scope: str, interval: int,
                            agg_type: str = "mean") -> List[Tuple[float, float]]:
        """Get aggregated values for a specific metric."""
        if interval not in self.aggregation_intervals:
            raise ValueError(f"Invalid aggregation interval: {interval}")
            
        metric_name = f"{name}_{agg_type}_{interval}s"
        
        if (scope not in self.aggregated_metrics[interval] or 
            metric_name not in self.aggregated_metrics[interval][scope]):
            return []
            
        return self.aggregated_metrics[interval][scope][metric_name]
    
    def get_latest_value(self, name: str, scope: str) -> Optional[Any]:
        """Get the most recent value for a metric."""
        if scope not in self.metrics or name not in self.metrics[scope]:
            return None
            
        values = self.metrics[scope][name]
        if not values:
            return None
            
        return values[-1][1]
    
    def export_metrics(self, filename: str = None) -> str:
        """Export all metrics to a file."""
        if not filename:
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.json"
            
        # Convert metrics to a serializable format
        serializable = {
            "raw_metrics": {
                scope: {
                    name: [{"timestamp": t, "value": v} for t, v in values]
                    for name, values in scope_metrics.items()
                }
                for scope, scope_metrics in self.metrics.items()
            },
            "aggregated_metrics": {
                str(interval): {
                    scope: {
                        name: [{"timestamp": t, "value": v} for t, v in values]
                        for name, values in scope_metrics.items()
                    }
                    for scope, scope_metrics in interval_metrics.items()
                }
                for interval, interval_metrics in self.aggregated_metrics.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
            
        logger.info(f"Exported metrics to {filename}")
        return filename


class StatisticalAnalyzer:
    """Performs statistical analysis on environment metrics."""
    
    def __init__(self, metrics_collector):
        """Initialize the statistical analyzer."""
        self.metrics_collector = metrics_collector
        
    def calculate_correlation(self, metric1: Tuple[str, str], 
                            metric2: Tuple[str, str],
                            window: int = None) -> float:
        """
        Calculate correlation between two metrics.
        
        Args:
            metric1: (scope, name) of the first metric
            metric2: (scope, name) of the second metric
            window: Optional number of most recent data points to use
            
        Returns:
            Correlation coefficient between the metrics
        """
        scope1, name1 = metric1
        scope2, name2 = metric2
        
        values1 = self.metrics_collector.get_metric_values(name1, scope1)
        values2 = self.metrics_collector.get_metric_values(name2, scope2)
        
        if not values1 or not values2:
            return 0.0
            
        # If window is specified, only use the most recent points
        if window:
            values1 = values1[-window:]
            values2 = values2[-window:]
            
        # Align the timestamps
        # This is a simple approach - for each point in series 1,
        # find the closest point in time from series 2
        aligned_values = []
        for t1, v1 in values1:
            if not isinstance(v1, (int, float)):
                continue
                
            # Find the closest timestamp in values2
            closest_idx = min(range(len(values2)), 
                            key=lambda i: abs(values2[i][0] - t1))
            t2, v2 = values2[closest_idx]
            
            # Only use if the value is numeric and the timestamps are close enough
            if isinstance(v2, (int, float)) and abs(t1 - t2) < 60:  # Within 60 seconds
                aligned_values.append((v1, v2))
                
        if len(aligned_values) < 2:
            return 0.0
            
        # Calculate correlation
        x_values = [v[0] for v in aligned_values]
        y_values = [v[1] for v in aligned_values]
        
        return np.corrcoef(x_values, y_values)[0, 1]
    
    def detect_anomalies(self, scope: str, name: str, 
                        method: str = "z_score", 
                        threshold: float = 3.0,
                        window: int = 100) -> List[Tuple[float, float]]:
        """
        Detect anomalies in a metric time series.
        
        Args:
            scope: The metric scope
            name: The metric name
            method: Detection method ('z_score', 'moving_avg', etc.)
            threshold: Threshold for anomaly detection
            window: Number of data points to use for detection
            
        Returns:
            List of (timestamp, value) tuples for detected anomalies
        """
        values = self.metrics_collector.get_metric_values(name, scope)
        
        # Filter only numeric values
        values = [(t, v) for t, v in values if isinstance(v, (int, float))]
        
        if len(values) < window:
            return []
            
        # Use only the most recent window of data
        recent_values = values[-window:]
        timestamps = [t for t, _ in recent_values]
        data_values = [v for _, v in recent_values]
        
        anomalies = []
        
        if method == "z_score":
            # Z-score anomaly detection
            mean = np.mean(data_values)
            std = np.std(data_values)
            
            if std == 0:
                return []
                
            for i, (t, v) in enumerate(recent_values):
                z_score = abs((v - mean) / std)
                if z_score > threshold:
                    anomalies.append((t, v))
                    
        elif method == "moving_avg":
            # Moving average anomaly detection
            window_size = min(20, len(data_values) // 4)
            
            for i in range(window_size, len(data_values)):
                window_mean = np.mean(data_values[i-window_size:i])
                window_std = np.std(data_values[i-window_size:i])
                
                if window_std == 0:
                    continue
                    
                z_score = abs((data_values[i] - window_mean) / window_std)
                if z_score > threshold:
                    anomalies.append((timestamps[i], data_values[i]))
                    
        return anomalies
    
    def calculate_trend(self, scope: str, name: str, 
                       window: int = None) -> Dict[str, float]:
        """
        Calculate trend statistics for a metric.
        
        Args:
            scope: The metric scope
            name: The metric name
            window: Optional number of most recent data points to use
            
        Returns:
            Dictionary with trend statistics
        """
        values = self.metrics_collector.get_metric_values(name, scope)
        
        # Filter only numeric values
        values = [(t, v) for t, v in values if isinstance(v, (int, float))]
        
        if len(values) < 2:
            return {
                "slope": 0.0,
                "direction": "stable",
                "strength": 0.0
            }
            
        # If window is specified, only use the most recent points
        if window:
            values = values[-window:]
            
        timestamps = np.array([t for t, _ in values])
        data_values = np.array([v for _, v in values])
        
        # Normalize timestamps to start from 0
        timestamps = timestamps - timestamps[0]
        
        # Simple linear regression
        try:
            slope, intercept = np.polyfit(timestamps, data_values, 1)
            
            # Calculate R-squared as an indicator of trend strength
            y_pred = slope * timestamps + intercept
            ss_total = np.sum((data_values - np.mean(data_values)) ** 2)
            ss_residual = np.sum((data_values - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            # Determine trend direction
            if abs(slope) < 1e-6:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
                
            return {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "direction": direction,
                "strength": abs(r_squared)
            }
        except Exception as e:
            logger.error(f"Error calculating trend for {scope}.{name}: {e}")
            return {
                "slope": 0.0,
                "direction": "stable",
                "strength": 0.0
            }


class TrendDetector:
    """Detects and monitors trends in environment metrics."""
    
    def __init__(self, metrics_collector, statistical_analyzer):
        """Initialize the trend detector."""
        self.metrics_collector = metrics_collector
        self.statistical_analyzer = statistical_analyzer
        self.active_trends = {}
        self.trend_history = []
        
    def detect_trends(self, scope: str = None, min_strength: float = 0.7) -> Dict[str, Any]:
        """
        Detect trends across all metrics or within a specific scope.
        
        Args:
            scope: Optional scope to limit trend detection
            min_strength: Minimum trend strength to report (R-squared value)
            
        Returns:
            Dictionary of detected trends by metric
        """
        trends = {}
        
        # Determine which metrics to analyze
        metrics_to_check = []
        for s in self.metrics_collector.metrics:
            if scope and s != scope and not s.startswith(f"{scope}."):
                continue
                
            for name in self.metrics_collector.metrics[s]:
                metrics_to_check.append((s, name))
                
        # Analyze each metric
        for s, name in metrics_to_check:
            try:
                # Calculate trend
                trend = self.statistical_analyzer.calculate_trend(s, name)
                
                # Filter based on strength
                if trend["strength"] >= min_strength:
                    trends[(s, name)] = trend
                    
            except Exception as e:
                logger.error(f"Error detecting trend for {s}.{name}: {e}")
                
        return trends
    
    def update_active_trends(self) -> None:
        """Update the set of active trends."""
        current_trends = self.detect_trends()
        current_time = time.time()
        
        # Update existing trends or add new ones
        for key, trend in current_trends.items():
            if key in self.active_trends:
                # Update existing trend
                self.active_trends[key]["current"] = trend
                self.active_trends[key]["last_updated"] = current_time
            else:
                # Add new trend
                self.active_trends[key] = {
                    "scope": key[0],
                    "name": key[1],
                    "first_detected": current_time,
                    "last_updated": current_time,
                    "initial": trend,
                    "current": trend
                }
                
        # Check for trends that have ended
        keys_to_remove = []
        for key, active_trend in self.active_trends.items():
            if key not in current_trends:
                # Trend has ended, move to history
                self.trend_history.append({
                    "scope": active_trend["scope"],
                    "name": active_trend["name"],
                    "first_detected": active_trend["first_detected"],
                    "last_updated": active_trend["last_updated"],
                    "duration": active_trend["last_updated"] - active_trend["first_detected"],
                    "initial": active_trend["initial"],
                    "final": active_trend["current"]
                })
                keys_to_remove.append(key)
                
        # Remove ended trends
        for key in keys_to_remove:
            del self.active_trends[key]
    
    def get_active_trends(self, min_duration: float = 0) -> List[Dict[str, Any]]:
        """
        Get currently active trends.
        
        Args:
            min_duration: Minimum trend duration in seconds
            
        Returns:
            List of active trend information
        """
        current_time = time.time()
        
        # Filter by duration
        return [
            {
                "scope": trend["scope"],
                "name": trend["name"],
                "duration": current_time - trend["first_detected"],
                "direction": trend["current"]["direction"],
                "strength": trend["current"]["strength"],
                "slope": trend["current"]["slope"]
            }
            for key, trend in self.active_trends.items()
            if current_time - trend["first_detected"] >= min_duration
        ]
    
    def get_trend_history(self) -> List[Dict[str, Any]]:
        """Get historical trends that are no longer active."""
        return self.trend_history


class ReportGenerator:
    """Generates reports from src.environment.env_base observations and metrics."""
    
    def __init__(self, observation_system, metrics_collector=None, 
                statistical_analyzer=None, trend_detector=None):
        """Initialize the report generator."""
        self.observation_system = observation_system
        self.metrics_collector = metrics_collector
        self.statistical_analyzer = statistical_analyzer
        self.trend_detector = trend_detector
        self.report_templates = {}
        self.scheduled_reports = {}
        self.report_history = []
        
    def register_template(self, name: str, template: Dict[str, Any]) -> None:
        """Register a report template."""
        self.report_templates[name] = template
        logger.info(f"Registered report template: {name}")
        
    def schedule_report(self, template_name: str, interval: int, 
                       params: Dict[str, Any] = None) -> str:
        """
        Schedule a report to be generated periodically.
        
        Args:
            template_name: Name of the template to use
            interval: Time interval between reports (in seconds)
            params: Parameters for the report
            
        Returns:
            ID of the scheduled report
        """
        if template_name not in self.report_templates:
            raise ValueError(f"Unknown report template: {template_name}")
            
        report_id = f"report_{template_name}_{int(time.time())}"
        
        self.scheduled_reports[report_id] = {
            "template_name": template_name,
            "interval": interval,
            "params": params or {},
            "next_run": time.time() + interval,
            "last_run": None
        }
        
        logger.info(f"Scheduled report {report_id} to run every {interval} seconds")
        return report_id
    
    def cancel_scheduled_report(self, report_id: str) -> None:
        """Cancel a scheduled report."""
        if report_id in self.scheduled_reports:
            del self.scheduled_reports[report_id]
            logger.info(f"Cancelled scheduled report: {report_id}")
            
    def check_scheduled_reports(self) -> None:
        """Check and run any scheduled reports that are due."""
        current_time = time.time()
        
        for report_id, config in list(self.scheduled_reports.items()):
            if current_time >= config["next_run"]:
                try:
                    # Generate the report
                    report = self.generate_report(
                        config["template_name"], config["params"]
                    )
                    
                    # Add to history
                    self.report_history.append({
                        "report_id": report_id,
                        "timestamp": current_time,
                        "report": report
                    })
                    
                    # Update next run time
                    self.scheduled_reports[report_id]["next_run"] = current_time + config["interval"]
                    self.scheduled_reports[report_id]["last_run"] = current_time
                    
                    logger.info(f"Generated scheduled report: {report_id}")
                except Exception as e:
                    logger.error(f"Error generating scheduled report {report_id}: {e}")
    
    def generate_report(self, template_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a report using the specified template.
        
        Args:
            template_name: Name of the template to use
            params: Parameters for the report
            
        Returns:
            Generated report
        """
        if template_name not in self.report_templates:
            raise ValueError(f"Unknown report template: {template_name}")
            
        template = self.report_templates[template_name]
        params = params or {}
        
        # Start with basic report info
        report = {
            "title": template.get("title", f"Report {template_name}"),
            "generated_at": time.time(),
            "generated_at_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "template": template_name,
            "sections": []
        }
        
        # Process each section in the template
        for section_config in template.get("sections", []):
            section_type = section_config.get("type")
            section = {"title": section_config.get("title", "Untitled Section"), "type": section_type}
            
            # Generate content based on section type
            if section_type == "observations":
                section["content"] = self._generate_observations_section(section_config, params)
            elif section_type == "metrics":
                section["content"] = self._generate_metrics_section(section_config, params)
            elif section_type == "trends":
                section["content"] = self._generate_trends_section(section_config, params)
            elif section_type == "anomalies":
                section["content"] = self._generate_anomalies_section(section_config, params)
            elif section_type == "summary":
                section["content"] = self._generate_summary_section(section_config, params)
            elif section_type == "custom":
                section["content"] = section_config.get("content", {})
                
            report["sections"].append(section)
            
        return report
    
    def _generate_observations_section(self, config: Dict[str, Any], 
                                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for an observations section."""
        obs_type = config.get("observation_type")
        env_id = config.get("environment_id") or params.get("environment_id")
        agent_id = config.get("agent_id") or params.get("agent_id")
        limit = config.get("limit", 10)
        
        observations = self.observation_system.get_recent_observations(
            count=limit, obs_type=obs_type, env_id=env_id, agent_id=agent_id
        )
        
        return {
            "observations": [
                {
                    "timestamp": obs.timestamp,
                    "timestamp_readable": datetime.fromtimestamp(obs.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                    "environment_id": obs.environment_id,
                    "observation_type": obs.observation_type,
                    "agent_id": obs.agent_id,
                    "sequence_id": obs.sequence_id,
                    "data": obs.data,
                    "tags": list(obs.tags)
                }
                for obs in observations
            ],
            "count": len(observations)
        }
    
    def _generate_metrics_section(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a metrics section."""
        if not self.metrics_collector:
            return {"error": "Metrics collector not available"}
            
        metrics = []
        scope_pattern = config.get("scope") or params.get("scope")
        name_pattern = config.get("name") or params.get("name")
        aggregation = config.get("aggregation", "raw")  # raw, 1min, 5min, etc.
        limit = config.get("limit", 100)
        
        # Determine which metrics to include
        for scope in self.metrics_collector.metrics:
            # Skip if doesn't match scope pattern
            if scope_pattern and not (scope == scope_pattern or scope.startswith(f"{scope_pattern}.")):
                continue
                
            for name in self.metrics_collector.metrics[scope]:
                # Skip if doesn't match name pattern
                if name_pattern and name != name_pattern and name_pattern not in name:
                    continue
                    
                # Get the metric values
                if aggregation == "raw":
                    values = self.metrics_collector.get_metric_values(name, scope)
                    values = values[-limit:] if limit else values
                    latest = values[-1][1] if values else None
                    
                    metrics.append({
                        "scope": scope,
                        "name": name,
                        "latest_value": latest,
                        "values": [
                            {"timestamp": t, "value": v}
                            for t, v in values
                        ]
                    })
                else:
                    # Handle aggregated metrics
                    if aggregation == "1min":
                        interval = 60
                    elif aggregation == "5min":
                        interval = 300
                    elif aggregation == "15min":
                        interval = 900
                    elif aggregation == "1hr":
                        interval = 3600
                    else:
                        continue
                        
                    values = self.metrics_collector.get_aggregated_values(
                        name, scope, interval, "mean"
                    )
                    values = values[-limit:] if limit else values
                    latest = values[-1][1] if values else None
                    
                    metrics.append({
                        "scope": scope,
                        "name": name,
                        "aggregation": aggregation,
                        "latest_value": latest,
                        "values": [
                            {"timestamp": t, "value": v}
                            for t, v in values
                        ]
                    })
        
        return {
            "metrics": metrics,
            "count": len(metrics)
        }
        
    def _generate_trends_section(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a trends section."""
        if not self.trend_detector:
            return {"error": "Trend detector not available"}
            
        include_active = config.get("include_active", True)
        include_history = config.get("include_history", True)
        min_strength = config.get("min_strength", 0.7)
        min_duration = config.get("min_duration", 0)
        
        result = {}
        
        if include_active:
            active_trends = self.trend_detector.get_active_trends(min_duration)
            # Filter by strength
            active_trends = [t for t in active_trends if t["strength"] >= min_strength]
            result["active_trends"] = active_trends
            result["active_count"] = len(active_trends)
            
        if include_history:
            history = self.trend_detector.get_trend_history()
            # Filter by strength
            history = [t for t in history if t["final"]["strength"] >= min_strength]
            result["trend_history"] = history
            result["history_count"] = len(history)
            
        return result
        
    def _generate_anomalies_section(self, config: Dict[str, Any],
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for an anomalies section."""
        if not self.statistical_analyzer:
            return {"error": "Statistical analyzer not available"}
            
        scope_pattern = config.get("scope") or params.get("scope")
        name_pattern = config.get("name") or params.get("name")
        method = config.get("method", "z_score")
        threshold = config.get("threshold", 3.0)
        window = config.get("window", 100)
        
        anomalies = []
        
        # Determine which metrics to analyze
        for scope in self.metrics_collector.metrics:
            # Skip if doesn't match scope pattern
            if scope_pattern and not (scope == scope_pattern or scope.startswith(f"{scope_pattern}.")):
                continue
                
            for name in self.metrics_collector.metrics[scope]:
                # Skip if doesn't match name pattern
                if name_pattern and name != name_pattern and name_pattern not in name:
                    continue
                    
                # Detect anomalies
                try:
                    detected = self.statistical_analyzer.detect_anomalies(
                        scope, name, method, threshold, window
                    )
                    
                    if detected:
                        anomalies.append({
                            "scope": scope,
                            "name": name,
                            "count": len(detected),
                            "anomalies": [
                                {"timestamp": t, "value": v}
                                for t, v in detected
                            ]
                        })
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {scope}.{name}: {e}")
        
        return {
            "anomalies": anomalies,
            "total_count": sum(a["count"] for a in anomalies),
            "metric_count": len(anomalies)
        }
        
    def _generate_summary_section(self, config: Dict[str, Any],
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a summary section."""
        result = {
            "timestamp": time.time(),
            "timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add environment summary
        if hasattr(self.observation_system, "registered_environments"):
            environments = list(self.observation_system.registered_environments.keys())
            result["environments"] = environments
            result["environment_count"] = len(environments)
            
        # Add agent summary if available
        if hasattr(self.observation_system, "agent_tracker") and self.observation_system.agent_tracker:
            agents = list(self.observation_system.agent_tracker.tracked_agents.keys())
            result["agents"] = agents
            result["agent_count"] = len(agents)
            
        # Add observation summary
        result["observation_count"] = len(self.observation_system.observations)
        
        # Add metrics summary if available
        if self.metrics_collector:
            metric_count = sum(
                len(metrics) for metrics in self.metrics_collector.metrics.values()
            )
            result["metric_count"] = metric_count
            
        # Add trend summary if available
        if self.trend_detector:
            active_trends = self.trend_detector.get_active_trends()
            result["active_trend_count"] = len(active_trends)
            
        return result
        
    def export_report(self, report: Dict[str, Any], format: str = "json",
                     filename: str = None) -> str:
        """
        Export a report to a file.
        
        Args:
            report: The report to export
            format: Export format (json, html, etc.)
            filename: Optional filename
            
        Returns:
            The filename of the exported report
        """
        if not filename:
            timestamp = int(time.time())
            filename = f"report_{timestamp}.{format}"
            
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        if format == "json":
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
                
        # In a real implementation, we would add support for other formats
        # like HTML, PDF, etc. here
                
        logger.info(f"Exported report to {filename}")
        return filename


class AlertSystem:
    """System for generating and managing alerts based on observations and metrics."""
    
    def __init__(self, observation_system, metrics_collector=None, risk_manager=None):
        """Initialize the alert system."""
        self.observation_system = observation_system
        self.metrics_collector = metrics_collector
        self.risk_manager = risk_manager
        self.alerts = []
        self.active_alerts = {}
        self.alert_rules = {}
        self.callbacks = {}
        self.alert_levels = ["info", "warning", "error", "critical"]
        
        # Register for relevant observations
        self.observation_system.register_callback(
            "environment_state", self._check_environment_alerts
        )
        if self.risk_manager:
            self.observation_system.register_callback(
                "risk_event", self._on_risk_event
            )
            
    def register_alert_rule(self, name: str, rule_config: Dict[str, Any]) -> None:
        """
        Register a new alert rule.
        
        Args:
            name: The name of the rule
            rule_config: Configuration for the rule
        """
        self.alert_rules[name] = rule_config
        logger.info(f"Registered alert rule: {name}")
        
    def register_callback(self, alert_level: str, callback: Callable) -> None:
        """Register a callback for a specific alert level."""
        if alert_level not in self.alert_levels:
            raise ValueError(f"Invalid alert level: {alert_level}")
            
        if alert_level not in self.callbacks:
            self.callbacks[alert_level] = []
            
        self.callbacks[alert_level].append(callback)
        
    def create_alert(self, level: str, source: str, message: str,
                   details: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a new alert.
        
        Args:
            level: Alert level (info, warning, error, critical)
            source: Source of the alert
            message: Alert message
            details: Additional details
            
        Returns:
            The created alert
        """
        if level not in self.alert_levels:
            raise ValueError(f"Invalid alert level: {level}")
            
        alert_id = f"alert_{int(time.time())}_{len(self.alerts)}"
        
        alert = {
            "id": alert_id,
            "level": level,
            "source": source,
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
            "timestamp_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "acknowledged": False
        }
        
        self.alerts.append(alert)
        
        # For error and critical alerts, mark as active
        if level in ["error", "critical"]:
            self.active_alerts[alert_id] = alert
            
        # Process callbacks
        if level in self.callbacks:
            for callback in self.callbacks[level]:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
        return alert
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: The ID of the alert to acknowledge
            
        Returns:
            True if acknowledged, False if not found
        """
        # Check active alerts
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]["acknowledged"] = True
            del self.active_alerts[alert_id]
            logger.info(f"Acknowledged active alert: {alert_id}")
            return True
            
        # Check in all alerts
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                logger.info(f"Acknowledged alert: {alert_id}")
                return True
                
        logger.warning(f"Attempt to acknowledge unknown alert: {alert_id}")
        return False
        
    def get_active_alerts(self, level: str = None) -> List[Dict[str, Any]]:
        """
        Get active alerts, optionally filtered by level.
        
        Args:
            level: Optional alert level to filter by
            
        Returns:
            List of active alerts
        """
        if level:
            return [a for a in self.active_alerts.values() if a["level"] == level]
        return list(self.active_alerts.values())
        
    def get_alerts(self, limit: int = None, level: str = None,
                 source: str = None) -> List[Dict[str, Any]]:
        """
        Get alerts, optionally filtered.
        
        Args:
            limit: Maximum number of alerts to return
            level: Optional alert level to filter by
            source: Optional source to filter by
            
        Returns:
            List of alerts
        """
        filtered = self.alerts
        
        if level:
            filtered = [a for a in filtered if a["level"] == level]
        if source:
            filtered = [a for a in filtered if a["source"] == source]
            
        # Return most recent first
        filtered = sorted(filtered, key=lambda a: a["timestamp"], reverse=True)
        
        if limit:
            filtered = filtered[:limit]
            
        return filtered
        
    def _check_environment_alerts(self, observation) -> None:
        """Check for environment alerts based on the observation."""
        if not self.metrics_collector:
            return
            
        # Check each alert rule
        for rule_name, rule_config in self.alert_rules.items():
            rule_type = rule_config.get("type")
            
            if rule_type == "threshold":
                # Threshold-based alert
                self._check_threshold_rule(rule_name, rule_config, observation)
            elif rule_type == "change":
                # Change-based alert
                self._check_change_rule(rule_name, rule_config, observation)
                
    def _check_threshold_rule(self, rule_name: str, rule_config: Dict[str, Any],
                           observation) -> None:
        """Check a threshold-based alert rule."""
        metric_scope = rule_config.get("metric_scope")
        metric_name = rule_config.get("metric_name")
        threshold = rule_config.get("threshold")
        operator = rule_config.get("operator", ">=")
        level = rule_config.get("level", "warning")
        
        if not metric_scope or not metric_name or threshold is None:
            return
            
        # Get the latest metric value
        latest_value = self.metrics_collector.get_latest_value(metric_name, metric_scope)
        
        if latest_value is None:
            return
            
        # Check threshold condition
        condition_met = False
        if operator == ">=":
            condition_met = latest_value >= threshold
        elif operator == ">":
            condition_met = latest_value > threshold
        elif operator == "<=":
            condition_met = latest_value <= threshold
        elif operator == "<":
            condition_met = latest_value < threshold
        elif operator == "==":
            condition_met = latest_value == threshold
        elif operator == "!=":
            condition_met = latest_value != threshold
            
        if condition_met:
            # Create alert
            self.create_alert(
                level=level,
                source=f"metric:{metric_scope}.{metric_name}",
                message=f"Metric {metric_name} {operator} {threshold} ({latest_value})",
                details={
                    "rule": rule_name,
                    "metric_scope": metric_scope,
                    "metric_name": metric_name,
                    "threshold": threshold,
                    "operator": operator,
                    "value": latest_value,
                    "observation_id": observation.sequence_id
                }
            )
            
    def _check_change_rule(self, rule_name: str, rule_config: Dict[str, Any],
                         observation) -> None:
        """Check a change-based alert rule."""
        metric_scope = rule_config.get("metric_scope")
        metric_name = rule_config.get("metric_name")
        change_threshold = rule_config.get("change_threshold")
        window = rule_config.get("window", 5)
        level = rule_config.get("level", "warning")
        
        if not metric_scope or not metric_name or change_threshold is None:
            return
            
        # Get recent metric values
        values = self.metrics_collector.get_metric_values(metric_name, metric_scope)
        
        if len(values) < window:
            return
            
        # Get the most recent values
        recent_values = [v for _, v in values[-window:]]
        
        # Calculate change
        if not all(isinstance(v, (int, float)) for v in recent_values):
            return
            
        first_value = recent_values[0]
        last_value = recent_values[-1]
        
        if first_value == 0:
            # Avoid division by zero
            change_pct = float('inf') if last_value > 0 else 0
        else:
            change_pct = abs((last_value - first_value) / first_value) * 100
            
        if change_pct >= change_threshold:
            # Create alert
            self.create_alert(
                level=level,
                source=f"metric:{metric_scope}.{metric_name}",
                message=f"Metric {metric_name} changed by {change_pct:.2f}% (threshold: {change_threshold}%)",
                details={
                    "rule": rule_name,
                    "metric_scope": metric_scope,
                    "metric_name": metric_name,
                    "change_threshold": change_threshold,
                    "change_pct": change_pct,
                    "first_value": first_value,
                    "last_value": last_value,
                    "window": window,
                    "observation_id": observation.sequence_id
                }
            )
            
    def _on_risk_event(self, observation) -> None:
        """Handle risk events from the risk manager."""
        data = observation.data
        
        if "level" not in data or "description" not in data:
            return
            
        risk_level = data["level"]
        description = data["description"]
        
        # Map risk level to alert level
        if risk_level == RiskLevel.LOW:
            alert_level = "info"
        elif risk_level == RiskLevel.MEDIUM:
            alert_level = "warning"
        elif risk_level == RiskLevel.HIGH:
            alert_level = "error"
        elif risk_level == RiskLevel.CRITICAL:
            alert_level = "critical"
        else:
            alert_level = "warning"
            
        # Create alert
        self.create_alert(
            level=alert_level,
            source="risk_manager",
            message=description,
            details=data
        )


class ObservationSystemManager:
    """Main manager class for the observation system and its components."""
    
    def __init__(self):
        """Initialize the observation system manager."""
        # Create the core observation system
        self.observation_system = create_default_observation_system()
        
        # Create component systems
        self.agent_tracker = AgentTracker(self.observation_system)
        self.environment_tracker = EnvironmentTracker(self.observation_system)
        self.interaction_tracker = InteractionTracker(self.observation_system)
        self.metrics_collector = MetricsCollector(self.observation_system)
        self.statistical_analyzer = StatisticalAnalyzer(self.metrics_collector)
        self.trend_detector = TrendDetector(self.metrics_collector, self.statistical_analyzer)
        self.report_generator = ReportGenerator(
            self.observation_system, 
            self.metrics_collector,
            self.statistical_analyzer,
            self.trend_detector
        )
        self.alert_system = AlertSystem(
            self.observation_system,
            self.metrics_collector
        )
        
        # Start monitoring threads
        self.monitoring_thread = None
        self.is_running = False
        
    def register_environment(self, env: Environment) -> None:
        """Register an environment with all components."""
        self.observation_system.register_environment(env)
        self.environment_tracker.register_environment(env)
        
        # Register default metrics for the environment
        if hasattr(env, "get_default_metrics"):
            default_metrics = env.get_default_metrics()
            for metric in default_metrics:
                self.metrics_collector.register_metric(
                    metric["name"],
                    f"environment.{env.env_id}",
                    metric.get("config")
                )
                
        logger.info(f"Registered environment {env.env_id} with all observation components")
        
    def register_agent(self, agent) -> None:
        """Register an agent with all components."""
        self.agent_tracker.register_agent(agent)
        
        # Register default metrics for the agent
        if hasattr(agent, "get_default_metrics"):
            default_metrics = agent.get_default_metrics()
            for metric in default_metrics:
                self.metrics_collector.register_metric(
                    metric["name"],
                    f"agent.{agent.id}",
                    metric.get("config")
                )
                
        logger.info(f"Registered agent {agent.id} with observation components")
        
    def start(self) -> None:
        """Start all observation system components."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start observation system
        self.observation_system.start_monitoring()
        
        # Start environment tracking
        self.environment_tracker.start_recording()
        
        # Start metrics collection and aggregation
        self.metrics_collector.start_aggregation()
        
        # Start monitoring thread for other periodic tasks
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Started observation system manager")
        
    def stop(self) -> None:
        """Stop all observation system components."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop observation system
        self.observation_system.stop_monitoring()
        
        # Stop environment tracking
        self.environment_tracker.stop_recording()
        
        # Stop metrics collection
        self.metrics_collector.stop_aggregation()
        
        # Stop monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Stopped observation system manager")
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for periodic tasks."""
        while self.is_running:
            try:
                # Update trend detection
                self.trend_detector.update_active_trends()
                
                # Check scheduled reports
                self.report_generator.check_scheduled_reports()
                
                # Other periodic tasks can be added here
            except Exception as e:
                logger.error(f"Error in observation system monitoring loop: {e}")
                
            # Sleep for a while
            time.sleep(10.0)
            
    def get_observation_summary(self) -> Dict[str, Any]:
        """Get a summary of the observation system state."""
        return {
            "is_running": self.is_running,
            "observation_count": len(self.observation_system.observations),
            "environments": list(self.observation_system.registered_environments.keys()),
            "agents": list(self.agent_tracker.tracked_agents.keys()),
            "active_alerts": len(self.alert_system.active_alerts),
            "active_trends": len(self.trend_detector.active_trends),
            "last_observation_time": self.observation_system.last_observation_time,
            "last_observation_time_readable": datetime.fromtimestamp(
                self.observation_system.last_observation_time
            ).strftime("%Y-%m-%d %H:%M:%S") if self.observation_system.last_observation_time else None
        }
        
    def export_all_data(self, base_dir: str = "export") -> Dict[str, str]:
        """Export all observation system data."""
        os.makedirs(base_dir, exist_ok=True)
        
        timestamp = int(time.time())
        files = {}
        
        # Export observations
        obs_file = os.path.join(base_dir, f"observations_{timestamp}.json")
        files["observations"] = self.observation_system.persist_observations(obs_file)
        
        # Export metrics
        if hasattr(self.metrics_collector, "export_metrics"):
            metrics_file = os.path.join(base_dir, f"metrics_{timestamp}.json")
            files["metrics"] = self.metrics_collector.export_metrics(metrics_file)
            
        # Export alerts
        alerts_file = os.path.join(base_dir, f"alerts_{timestamp}.json")
        with open(alerts_file, 'w') as f:
            json.dump({
                "active_alerts": self.alert_system.get_active_alerts(),
                "all_alerts": self.alert_system.get_alerts()
            }, f, indent=2)
        files["alerts"] = alerts_file
        
        # Export summary report
        summary_file = os.path.join(base_dir, f"summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(self.get_observation_summary(), f, indent=2)
        files["summary"] = summary_file
        
        logger.info(f"Exported all observation system data to {base_dir}")
        return files

# Utility functions needed for env_base.py

def extract_observation(df, start_idx, end_idx, available_cols):
    """
    Extract raw observation from dataframe.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data.
        start_idx (int): Start index of observation window.
        end_idx (int): End index of observation window.
        available_cols (list): Available columns in dataframe.
        
    Returns:
        dict: Raw observation data
    """
    # Handle errors gracefully
    if df is None:
        logger.error("Cannot extract observation: DataFrame is None")
        return None
        
    if start_idx < 0 or end_idx >= len(df) or start_idx > end_idx:
        logger.error(f"Invalid indices for observation extraction: start={start_idx}, end={end_idx}, df_len={len(df)}")
        return None
    
    # Get slice of dataframe
    try:
        window = df.iloc[start_idx:end_idx+1]
    except Exception as e:
        logger.error(f"Error slicing dataframe: {e}")
        return None
    
    # Make sure we have at least close price
    if 'close' not in window.columns:
        logger.error("'close' column missing from DataFrame - required for observation")
        return None
    
    # Get last row for current information
    current = window.iloc[-1]
    
    # Get close prices for returns calculation
    closes = window['close'].values
    
    # Calculate returns
    returns = np.diff(closes) / closes[:-1] if len(closes) > 1 else np.array([0.0])
    
    # Calculate volatility
    volatility = np.std(returns) if len(returns) > 1 else 0.0
    
    # Extract features present in the dataframe
    features = {}
    missing_features = []
    
    for col in available_cols:
        if col in window.columns:
            try:
                # Get values and ensure they're numeric
                values = window[col].values
                if np.isnan(values).any():
                    # Handle NaN values by filling with forward fill, then backward fill
                    values = pd.Series(values).fillna(method='ffill').fillna(method='bfill').values
                features[col] = values
            except Exception as e:
                logger.warning(f"Error extracting feature '{col}': {e}")
                missing_features.append(col)
        else:
            missing_features.append(col)
    
    # Log any missing features
    if missing_features:
        logger.warning(f"Some requested features were not available: {missing_features}")
    
    # Get current price and time info
    current_price = current['close']
    
    # Create observation
    observation = {
        'window': window,
        'features': features,
        'current_price': current_price,
        'returns': returns,
        'volatility': volatility,
        'feature_count': len(features)
    }
    
    return observation

def standardize_observation(observation, agent=None):
    """
    Standardize observation to consistent format.
    
    Args:
        observation (dict): Raw observation data.
        agent (Any, optional): Agent object with positions and state.
        
    Returns:
        dict: Standardized observation
    """
    # If no observation, return None
    if observation is None:
        return None
        
    window = observation['window']
    features = observation['features']
    current_price = observation['current_price']
    
    # Extract market data
    market_data = {
        'price': current_price,
        'returns': observation['returns'],
        'volatility': observation['volatility']
    }
    
    # Add available technical indicators
    for key, value in features.items():
        if key not in ['open', 'high', 'low', 'close', 'volume']:
            market_data[key] = value
    
    # Initialize position and agent state components
    position_data = {}
    agent_state = {}
    
    if agent:
        # Get position information
        positions = getattr(agent, 'positions', [])
        position_data = {
            'num_positions': len(positions),
            'avg_entry_price': np.mean([p.entry_price for p in positions]) if positions else 0,
            'unrealized_pnl': sum([(current_price - p.entry_price) * p.size for p in positions]),
            'position_sizes': [p.size for p in positions],
            'position_entries': [p.entry_price for p in positions],
            'position_ages': [p.bars_held for p in positions] if hasattr(positions[0], 'bars_held') else []
        }
        
        # Get agent state
        capital = getattr(agent, 'capital', 0)
        initial_capital = getattr(agent, 'initial_capital', 1)
        
        agent_state = {
            'capital': capital,
            'initial_capital': initial_capital,
            'capital_ratio': capital / initial_capital if initial_capital > 0 else 1.0,
        }
        
        # Add withdrawal-related information
        withdrawals = getattr(agent, 'withdrawals', [])
        usdt_balance = getattr(agent, 'usdt_balance', capital)
        usd_balance = getattr(agent, 'usd_balance', 0)
        usd_reserved = getattr(agent, 'usd_reserved', 0)
        
        active_withdrawals = [w for w in withdrawals 
                              if w.status.value in [0, 1]]  # PENDING or PARTIAL
        
        withdrawal_info = {
            'has_withdrawals': len(active_withdrawals) > 0,
            'num_withdrawals': len(active_withdrawals),
            'total_withdrawal_amount': sum(w.get_remaining_amount() for w in active_withdrawals),
            'max_urgency': max([w.get_urgency() for w in active_withdrawals], default=0),
            'avg_urgency': np.mean([w.get_urgency() for w in active_withdrawals]) if active_withdrawals else 0,
            'emergency_count': sum(1 for w in active_withdrawals if w.withdrawal_type.value == 1),  # EMERGENCY
            'timed_count': sum(1 for w in active_withdrawals if w.withdrawal_type.value == 0),  # TIMED
            'standard_count': sum(1 for w in active_withdrawals if w.withdrawal_type.value == 2),  # STANDARD
            'usd_balance': usd_balance,
            'usd_reserved': usd_reserved,
            'usdt_balance': usdt_balance,
            'withdrawals_to_capital_ratio': sum(w.get_remaining_amount() for w in active_withdrawals) / usdt_balance if usdt_balance > 0 else 0
        }
        
        # Add withdrawal info to agent state
        agent_state.update(withdrawal_info)
    
    # Combine all parts
    standardized_obs = {
        'market_data': market_data,
        'position_data': position_data,
        'agent_state': agent_state
    }
    
    return standardized_obs
