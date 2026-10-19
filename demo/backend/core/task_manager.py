from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from threading import RLock
from time import time
from uuid import uuid4


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    outputs: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time)


class TaskManager:
    def __init__(self):
        self._jobs = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ultralabel")

    def create_job(self):
        job = Job()
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def start(self, job, callback):
        self._executor.submit(self._run, job, callback)

    def get_job(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id):
        with self._lock:
            return asdict(self._jobs[job_id])

    def _run(self, job, callback):
        with self._lock:
            job.status = "running"
        try:
            callback(job)
            with self._lock:
                job.status, job.progress = "succeeded", 1.0
        except Exception as error:
            with self._lock:
                job.status, job.message = "failed", str(error)


task_manager = TaskManager()
