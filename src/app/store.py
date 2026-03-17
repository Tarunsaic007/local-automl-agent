import threading 
from dataclasses import dataclass,field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional



class JobStatus(str, Enum):
    QUEUED = "queued"
    ANALYZING = "analyzing"
    STRATEGIZING = "strategizing"
    TRAINING = "training"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.QUEDED
    progress: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
class JobStore:
    """Singleton, thread-safe job registry."""

    _instance: Optional["JobStore"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.rw_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "JobStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
    
    def create(self, job_id: str) -> Job:
        with self._rw_lock:
            job = Job(job_id=job_id)
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._rw_lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, status: JobStatus, progress: int = None) -> None:
        with self._rw_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if progress is not None:
                    job.progress = progress
                job.updated_at = datetime.now(timezone.utc).isoformat()

    def complete(self, job_id: str, result: dict) -> None:
        with self._rw_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETE
                job.progress = 100
                job.result = result
                job.updated_at = datetime.now(timezone.utc).isoformat()

    def fail(self, job_id: str, error: str) -> None:
        with self._rw_lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = error
                job.updated_at = datetime.now(timezone.utc).isoformat()

    def list_jobs(self) -> list[dict]:
        with self._rw_lock:
            return [j.to_dict() for j in self._jobs.values()]