"""Admin dashboard routes: system snapshot, metrics, ML health."""

import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from src.models import User, Assessment, RequestMetric
from src.middleware.deps import require_admin
from src.services.ml_client import MLClient

logger = logging.getLogger("mindscope")

router = APIRouter(prefix="/admin/dashboard", tags=["admin"])


@router.get("/snapshot")
async def admin_snapshot(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """System-wide dashboard data for admins."""
    # Totals (exclude admin users)
    total_users = (await db.execute(
        select(func.count(User.id)).where(User.role != "admin")
    )).scalar() or 0
    total_doctors = (await db.execute(
        select(func.count(User.id)).where(User.role == "doctor")
    )).scalar() or 0
    total_patients = (await db.execute(
        select(func.count(User.id)).where(User.role == "patient")
    )).scalar() or 0
    total_assessments = (await db.execute(
        select(func.count(Assessment.id))
    )).scalar() or 0

    # Recent users (exclude admin users)
    users_result = await db.execute(
        select(User).where(User.role != "admin").order_by(desc(User.created_at)).limit(100)
    )
    users_list = [
        {
            "id": u.id,
            "name": u.name,
            "email": u.email,
            "role": u.role,
            "age": u.age,
            "basicInfo": u.basic_info,
            "createdAt": u.created_at.isoformat() if u.created_at else None,
        }
        for u in users_result.scalars().all()
    ]

    # Recent assessments (with user info)
    assessments_result = await db.execute(
        select(Assessment).order_by(desc(Assessment.created_at)).limit(100)
    )
    assessments_list = []
    for a in assessments_result.scalars().all():
        # Get user for this assessment
        user_result = await db.execute(select(User).where(User.id == a.user_id))
        u = user_result.scalar_one_or_none()
        assessments_list.append({
            "id": a.id,
            "userId": a.user_id,
            "userName": u.name if u else "Unknown",
            "email": u.email if u else "",
            "score": a.score_total,
            "severity": a.severity,
            "recordingCount": a.recording_count,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
        })

    return {
        "totals": {
            "users": total_users,
            "doctors": total_doctors,
            "patients": total_patients,
            "assessments": total_assessments,
        },
        "users": users_list,
        "assessments": assessments_list,
    }


@router.get("/metrics")
async def admin_metrics(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Aggregated request metrics over last 24 hours in 5-min buckets."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    result = await db.execute(
        select(RequestMetric)
        .where(RequestMetric.created_at >= cutoff)
        .order_by(RequestMetric.created_at)
    )
    rows = result.scalars().all()

    # Bucket into 5-min windows
    buckets: dict[str, dict] = {}
    for r in rows:
        ts = r.created_at.replace(minute=(r.created_at.minute // 5) * 5, second=0, microsecond=0)
        key = ts.isoformat()
        if key not in buckets:
            buckets[key] = {"time": key, "count": 0, "total_latency": 0.0, "errors": 0}
        buckets[key]["count"] += 1
        buckets[key]["total_latency"] += r.latency_ms
        if r.status_code >= 400:
            buckets[key]["errors"] += 1

    timeline = []
    for b in buckets.values():
        timeline.append({
            "time": b["time"],
            "requests": b["count"],
            "avgLatency": round(b["total_latency"] / b["count"], 1) if b["count"] else 0,
            "errorRate": round(b["errors"] / b["count"] * 100, 1) if b["count"] else 0,
        })

    # ML severity distribution from recent assessments
    severity_result = await db.execute(
        select(Assessment.ml_severity, func.count(Assessment.id))
        .where(Assessment.ml_severity.isnot(None))
        .group_by(Assessment.ml_severity)
    )
    severity_dist = [{"name": s, "value": c} for s, c in severity_result.all()]

    # Recent ML predictions
    recent_result = await db.execute(
        select(Assessment)
        .where(Assessment.ml_score.isnot(None))
        .order_by(desc(Assessment.created_at))
        .limit(20)
    )
    recent = [
        {
            "id": a.id,
            "mlScore": a.ml_score,
            "mlSeverity": a.ml_severity,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
        }
        for a in recent_result.scalars().all()
    ]

    return {
        "timeline": timeline,
        "severityDistribution": severity_dist,
        "recentPredictions": recent,
    }


@router.get("/ml-health")
async def admin_ml_health(
    user: User = Depends(require_admin),
):
    """Check health of backend and ML model services."""
    import time

    # Backend health (we're alive if this runs)
    backend_status = {"status": "healthy", "latency_ms": 0}

    # ML model health
    ml_status = {"status": "unreachable", "latency_ms": 0}
    try:
        client = MLClient()
        t0 = time.perf_counter()
        health = await client.health_check()
        ml_status = {
            "status": health.get("status", "healthy"),
            "modelLoaded": health.get("model_loaded", False),
            "device": health.get("device", "unknown"),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as e:
        ml_status["error"] = str(e)

    return {
        "backend": backend_status,
        "mlModel": ml_status,
    }
