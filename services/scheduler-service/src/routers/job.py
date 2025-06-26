from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging
import uuid
import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)

class JobCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    job_type: str = Field(..., regex="^(reminder|recurring_task|notification|cleanup)$")
    schedule_time: datetime
    recurring: bool = False
    recurring_pattern: Optional[str] = None  # 'daily', 'weekly', 'monthly'
    payload: Dict[str, Any] = {}
    priority: str = Field(default="medium", regex="^(low|medium|high|urgent)$")

class JobResponse(BaseModel):
    id: str
    user_id: str
    name: str
    job_type: str
    status: str  # 'pending', 'running', 'completed', 'failed', 'cancelled'
    schedule_time: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    recurring: bool
    recurring_pattern: Optional[str]
    next_run: Optional[datetime]
    payload: Dict[str, Any]
    priority: str
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    created_at: datetime
    updated_at: datetime

class JobStats(BaseModel):
    total_jobs: int
    pending_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    jobs_today: int
    upcoming_jobs: List[Dict[str, Any]]

# Mock storage
jobs_storage = {}
job_queue = []

def get_current_user_id() -> str:
    return "user-123"

async def execute_job(job_id: str):
    """Execute a background job"""
    try:
        job = jobs_storage.get(job_id)
        if not job:
            return
        
        job["status"] = "running"
        job["started_at"] = datetime.utcnow()
        
        # Mock job execution based on type
        await asyncio.sleep(1)  # Simulate work
        
        if job["job_type"] == "reminder":
            logger.info(f"Sending reminder: {job['payload'].get('message', 'No message')}")
        elif job["job_type"] == "notification":
            logger.info(f"Sending notification: {job['payload'].get('message', 'No message')}")
        elif job["job_type"] == "cleanup":
            logger.info(f"Running cleanup task: {job['name']}")
        
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow()
        
        # Schedule next run for recurring jobs
        if job["recurring"] and job["recurring_pattern"]:
            schedule_next_run(job)
        
        logger.info(f"Job completed: {job_id}")
        
    except Exception as e:
        job["status"] = "failed"
        job["error_message"] = str(e)
        job["retry_count"] += 1
        logger.error(f"Job failed: {job_id} - {e}")

def schedule_next_run(job):
    """Schedule the next run for a recurring job"""
    if job["recurring_pattern"] == "daily":
        next_run = job["schedule_time"] + timedelta(days=1)
    elif job["recurring_pattern"] == "weekly":
        next_run = job["schedule_time"] + timedelta(weeks=1)
    elif job["recurring_pattern"] == "monthly":
        next_run = job["schedule_time"] + timedelta(days=30)
    else:
        return
    
    # Create new job instance
    new_job_id = str(uuid.uuid4())
    new_job = job.copy()
    new_job["id"] = new_job_id
    new_job["schedule_time"] = next_run
    new_job["status"] = "pending"
    new_job["started_at"] = None
    new_job["completed_at"] = None
    new_job["error_message"] = None
    new_job["retry_count"] = 0
    new_job["created_at"] = datetime.utcnow()
    new_job["updated_at"] = datetime.utcnow()
    
    jobs_storage[new_job_id] = new_job
    job["next_run"] = next_run

@router.post("/", response_model=JobResponse)
async def create_job(job_data: JobCreate, background_tasks: BackgroundTasks):
    """Create a new scheduled job"""
    try:
        job_id = str(uuid.uuid4())
        user_id = get_current_user_id()
        
        job = {
            "id": job_id,
            "user_id": user_id,
            "name": job_data.name,
            "job_type": job_data.job_type,
            "status": "pending",
            "schedule_time": job_data.schedule_time,
            "started_at": None,
            "completed_at": None,
            "recurring": job_data.recurring,
            "recurring_pattern": job_data.recurring_pattern,
            "next_run": None,
            "payload": job_data.payload,
            "priority": job_data.priority,
            "error_message": None,
            "retry_count": 0,
            "max_retries": 3,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        jobs_storage[job_id] = job
        
        # Schedule job execution if it's due soon (within 1 hour)
        if job_data.schedule_time <= datetime.utcnow() + timedelta(hours=1):
            delay = max(0, (job_data.schedule_time - datetime.utcnow()).total_seconds())
            background_tasks.add_task(execute_job_after_delay, job_id, delay)
        
        logger.info(f"Created job: {job_id} for user: {user_id}")
        
        return JobResponse(**job)
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail="Failed to create job")

async def execute_job_after_delay(job_id: str, delay: float):
    """Execute job after a delay"""
    await asyncio.sleep(delay)
    await execute_job(job_id)

@router.get("/", response_model=List[JobResponse])
async def get_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    job_type: Optional[str] = Query(None),
    from_date: Optional[datetime] = Query(None),
    to_date: Optional[datetime] = Query(None)
):
    """Get user's scheduled jobs"""
    try:
        user_id = get_current_user_id()
        user_jobs = [job for job in jobs_storage.values() if job["user_id"] == user_id]
        
        # Apply filters
        if status:
            user_jobs = [job for job in user_jobs if job["status"] == status]
        
        if job_type:
            user_jobs = [job for job in user_jobs if job["job_type"] == job_type]
        
        if from_date:
            user_jobs = [job for job in user_jobs if job["schedule_time"] >= from_date]
        
        if to_date:
            user_jobs = [job for job in user_jobs if job["schedule_time"] <= to_date]
        
        # Sort by schedule time
        user_jobs.sort(key=lambda x: x["schedule_time"])
        
        # Apply pagination
        paginated_jobs = user_jobs[skip:skip + limit]
        
        return [JobResponse(**job) for job in paginated_jobs]
        
    except Exception as e:
        logger.error(f"Error getting jobs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get jobs")

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get a specific job"""
    try:
        job = jobs_storage.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check ownership
        user_id = get_current_user_id()
        if job["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return JobResponse(**job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job")

@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    try:
        job = jobs_storage.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check ownership
        user_id = get_current_user_id()
        if job["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if job["status"] in ["pending", "running"]:
            job["status"] = "cancelled"
            job["updated_at"] = datetime.utcnow()
            
            logger.info(f"Job cancelled: {job_id}")
            return {"message": "Job cancelled successfully"}
        else:
            return {"message": "Job cannot be cancelled (already completed or failed)"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")

@router.post("/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed job"""
    try:
        job = jobs_storage.get(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check ownership
        user_id = get_current_user_id()
        if job["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if job["status"] == "failed" and job["retry_count"] < job["max_retries"]:
            job["status"] = "pending"
            job["error_message"] = None
            job["updated_at"] = datetime.utcnow()
            
            # Schedule immediate retry
            background_tasks.add_task(execute_job, job_id)
            
            logger.info(f"Job retry scheduled: {job_id}")
            return {"message": "Job retry scheduled"}
        else:
            return {"message": "Job cannot be retried (max retries reached or not failed)"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying job: {e}")
        raise HTTPException(status_code=500, detail="Failed to retry job")

@router.get("/stats/summary", response_model=JobStats)
async def get_job_stats():
    """Get job statistics"""
    try:
        user_id = get_current_user_id()
        user_jobs = [job for job in jobs_storage.values() if job["user_id"] == user_id]
        
        total_jobs = len(user_jobs)
        pending_jobs = len([j for j in user_jobs if j["status"] == "pending"])
        running_jobs = len([j for j in user_jobs if j["status"] == "running"])
        completed_jobs = len([j for j in user_jobs if j["status"] == "completed"])
        failed_jobs = len([j for j in user_jobs if j["status"] == "failed"])
        
        # Jobs today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        jobs_today = len([j for j in user_jobs if j["schedule_time"] >= today_start])
        
        # Upcoming jobs (next 24 hours)
        tomorrow = datetime.utcnow() + timedelta(hours=24)
        upcoming = [j for j in user_jobs 
                   if j["status"] == "pending" and j["schedule_time"] <= tomorrow]
        upcoming.sort(key=lambda x: x["schedule_time"])
        
        upcoming_jobs = [
            {
                "id": job["id"],
                "name": job["name"],
                "job_type": job["job_type"],
                "schedule_time": job["schedule_time"],
                "priority": job["priority"]
            }
            for job in upcoming[:10]  # Next 10 jobs
        ]
        
        return JobStats(
            total_jobs=total_jobs,
            pending_jobs=pending_jobs,
            running_jobs=running_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs,
            jobs_today=jobs_today,
            upcoming_jobs=upcoming_jobs
        )
        
    except Exception as e:
        logger.error(f"Error getting job stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job stats")
