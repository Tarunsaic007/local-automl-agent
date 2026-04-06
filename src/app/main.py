'''
Fast API application -  entry point for all HTTP requests

Endpoints:
POST /submit         → Upload CSV, returns job_id
  GET  /status/{id}   → Poll job progress and results
  GET  /jobs           → List all jobs
  GET  /health         → Health check

'''

from contextlib import asynccontextmanager

# from app import store
from fastapi import BackgroundTasks, FAstAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from logging import logger

from app.agent import start_job
from app.store import JobStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Startup and Shutdown events'''
    logger.info("starting Local AutoML Agent...")
    JobStore.get_instance()  # Initialize the JobStore singleton
    yield
    logger.info("shutting down Local AutoML Agent...")


app = FastAPI(
    title="Local AutoML Agent",
    description="Autonomous ML Pipeline Generator using local LLM orchestration",
    version="1.0.0",
    lifespan=lifespan
)

app.addmiddleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """ Bsdic health check - confirms service is alive."""
    return {"status": "ok", "service": "Local AutoML Agent"}

@app.post("/submit", status_code=202)
async def submit_dataset(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a CSV dataset to trigger an AutoML pipeline.
    Returns a job_id for pooling /status.
    """
    if not file.name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    job_id = start_job(content)
    logger.info(f"Jpb {job_id} submitted for file: {file.filename}")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Pipeline started.Po;; /status/{job_id} for updates.",
    }

@app.get("/status/{job_id}")
def get_status(job_id:str):
    """
    Poll the status of a submitted pipeline job. 
    Returns progress (0-100), status string, and result when complete."""

    store = JobStore.get_instance()
    job = store.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()

@app.get("/jobs")
def list_jobs():
    """List all jobs and their current statuses."""
    store = JobStore.get_instance()
    return {"jobs": [job.to_dict() for job in store.list_jobs()]}