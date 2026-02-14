"""
AACO-SIGMA Job Scheduler

Schedules profiling jobs across the GPU fleet.
Handles job queuing, execution, and result collection.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time
import uuid
from queue import PriorityQueue

from .fleet_manager import FleetManager, NodeStatus


class JobStatus(Enum):
    """Status of a profiling job."""

    PENDING = auto()  # In queue
    SCHEDULED = auto()  # Assigned to node
    RUNNING = auto()  # Currently executing
    COMPLETED = auto()  # Finished successfully
    FAILED = auto()  # Execution failed
    CANCELLED = auto()  # Cancelled by user
    TIMEOUT = auto()  # Exceeded time limit


class JobPriority(Enum):
    """Job priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class JobConfig:
    """Job execution configuration."""

    # Profiling options
    profile_type: str = "standard"  # standard, detailed, minimal
    collect_counters: bool = True
    collect_traces: bool = False

    # Execution
    iterations: int = 3
    warmup_iterations: int = 1
    timeout_s: int = 300

    # Requirements
    required_gfx: Optional[str] = None
    required_memory_gb: float = 0.0
    required_tags: List[str] = field(default_factory=list)


@dataclass
class ProfileJob:
    """A profiling job to execute."""

    # Identity
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""

    # What to profile
    workload_id: str = ""
    model_path: Optional[str] = None
    script_path: Optional[str] = None

    # Configuration
    config: JobConfig = field(default_factory=JobConfig)
    priority: JobPriority = JobPriority.NORMAL

    # Status
    status: JobStatus = JobStatus.PENDING
    assigned_node: Optional[str] = None

    # Timing
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2

    # Metadata
    requester: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def queue_time_s(self) -> float:
        if self.started_at and self.created_at:
            return self.started_at - self.created_at
        return 0.0

    @property
    def execution_time_s(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0


@dataclass
class JobResult:
    """Result of a profiling job."""

    job_id: str

    # Execution
    node_id: str = ""
    gpu_index: int = 0

    # Results
    success: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Timing results
    latency_ms: float = 0.0
    throughput: float = 0.0

    # Counter data
    counters: Dict[str, float] = field(default_factory=dict)

    # Trace data
    trace_file: Optional[str] = None

    # Errors
    error: Optional[str] = None


class JobScheduler:
    """
    Schedules and executes profiling jobs across fleet.

    Features:
    - Priority-based queuing
    - Node matching based on requirements
    - Automatic retry on failure
    - Result aggregation
    """

    def __init__(self, fleet_manager: FleetManager):
        self.fleet = fleet_manager
        self._queue: PriorityQueue = PriorityQueue()
        self._running_jobs: Dict[str, ProfileJob] = {}
        self._completed_jobs: Dict[str, JobResult] = {}

        # Callbacks
        self._completion_callbacks: List[Callable] = []

        # Stats
        self._jobs_submitted = 0
        self._jobs_completed = 0
        self._jobs_failed = 0

    def submit(self, job: ProfileJob) -> str:
        """
        Submit a job for execution.

        Args:
            job: Job to submit

        Returns:
            Job ID
        """
        self._jobs_submitted += 1

        # Enqueue with priority
        priority = (job.priority.value, time.time())
        self._queue.put((priority, job))

        return job.job_id

    def submit_batch(self, jobs: List[ProfileJob]) -> List[str]:
        """Submit multiple jobs."""
        return [self.submit(job) for job in jobs]

    def schedule_pending(self) -> int:
        """
        Schedule pending jobs to available nodes.

        Returns:
            Number of jobs scheduled
        """
        scheduled = 0

        while not self._queue.empty():
            # Check if we can schedule more
            available_nodes = self.fleet.get_available_nodes()
            if not available_nodes:
                break

            # Get next job
            priority, job = self._queue.get()

            # Find suitable node
            requirements = {
                "gfx_version": job.config.required_gfx,
                "min_memory_gb": job.config.required_memory_gb,
                "tags": job.config.required_tags,
            }

            node = self.fleet.select_node(requirements)

            if node:
                # Assign job
                job.status = JobStatus.SCHEDULED
                job.assigned_node = node.node_id
                job.scheduled_at = time.time()

                self._running_jobs[job.job_id] = job

                # Mark node as busy
                node.status = NodeStatus.BUSY

                scheduled += 1
            else:
                # Re-queue
                self._queue.put((priority, job))
                break

        return scheduled

    def start_job(self, job_id: str) -> bool:
        """
        Mark job as started.

        Called when execution actually begins on node.
        """
        if job_id not in self._running_jobs:
            return False

        job = self._running_jobs[job_id]
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        return True

    def complete_job(self, job_id: str, result: JobResult) -> bool:
        """
        Complete a job with results.
        """
        if job_id not in self._running_jobs:
            return False

        job = self._running_jobs.pop(job_id)

        if result.success:
            job.status = JobStatus.COMPLETED
            self._jobs_completed += 1
        else:
            job.status = JobStatus.FAILED
            self._jobs_failed += 1

            # Retry if possible
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.assigned_node = None
                priority = (job.priority.value, time.time())
                self._queue.put((priority, job))
                return True

        job.completed_at = time.time()
        self._completed_jobs[job_id] = result

        # Release node
        if job.assigned_node:
            node = self.fleet.get_node(job.assigned_node)
            if node:
                node.status = NodeStatus.ONLINE

        # Notify callbacks
        for callback in self._completion_callbacks:
            try:
                callback(job, result)
            except Exception:
                pass

        return True

    def fail_job(self, job_id: str, error: str) -> bool:
        """
        Mark job as failed.
        """
        result = JobResult(
            job_id=job_id,
            success=False,
            error=error,
        )
        return self.complete_job(job_id, result)

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.
        """
        if job_id in self._running_jobs:
            job = self._running_jobs.pop(job_id)
            job.status = JobStatus.CANCELLED

            # Release node
            if job.assigned_node:
                node = self.fleet.get_node(job.assigned_node)
                if node:
                    node.status = NodeStatus.ONLINE

            return True

        # Check queue (inefficient but works)
        new_queue = PriorityQueue()
        found = False

        while not self._queue.empty():
            priority, job = self._queue.get()
            if job.job_id == job_id:
                job.status = JobStatus.CANCELLED
                found = True
            else:
                new_queue.put((priority, job))

        self._queue = new_queue
        return found

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a job."""
        if job_id in self._running_jobs:
            return self._running_jobs[job_id].status
        if job_id in self._completed_jobs:
            return JobStatus.COMPLETED
        return None

    def get_result(self, job_id: str) -> Optional[JobResult]:
        """Get result of completed job."""
        return self._completed_jobs.get(job_id)

    def get_pending_count(self) -> int:
        """Get number of pending jobs."""
        return self._queue.qsize()

    def get_running_count(self) -> int:
        """Get number of running jobs."""
        return len(self._running_jobs)

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "jobs_pending": self._queue.qsize(),
            "jobs_running": len(self._running_jobs),
            "success_rate": self._jobs_completed
            / max(1, self._jobs_submitted + self._jobs_failed)
            * 100,
        }

    def register_completion_callback(self, callback: Callable) -> None:
        """Register callback for job completion."""
        self._completion_callbacks.append(callback)

    def check_timeouts(self) -> List[str]:
        """
        Check for timed out jobs.

        Returns:
            List of job IDs that timed out
        """
        timed_out = []
        current_time = time.time()

        for job_id, job in list(self._running_jobs.items()):
            if job.started_at:
                elapsed = current_time - job.started_at
                if elapsed > job.config.timeout_s:
                    job.status = JobStatus.TIMEOUT
                    self.fail_job(job_id, f"Job timed out after {elapsed:.0f}s")
                    timed_out.append(job_id)

        return timed_out
