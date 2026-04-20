"""Doctor dashboard routes: summary, alerts, patient trends."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from database import get_db
from src.models import User, Assessment
from src.middleware.deps import require_doctor

router = APIRouter(prefix="/doctor/dashboard", tags=["doctor"])


@router.get("/summary")
async def doctor_summary(
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate stats for the doctor dashboard."""
    # Total patients
    patients_count = (await db.execute(
        select(func.count(User.id)).where(User.role == "patient")
    )).scalar() or 0

    # Total assessments
    assessments_count = (await db.execute(
        select(func.count(Assessment.id))
    )).scalar() or 0

    # High risk (Severe + Moderately Severe)
    high_risk = (await db.execute(
        select(func.count(Assessment.id)).where(
            Assessment.severity.in_(["Severe", "Moderately Severe"])
        )
    )).scalar() or 0

    low_risk = (await db.execute(
        select(func.count(Assessment.id)).where(
            Assessment.severity.in_(["Minimal", "Mild"])
        )
    )).scalar() or 0

    # Severity breakdown
    breakdown_q = (
        select(Assessment.severity, func.count(Assessment.id).label("count"))
        .group_by(Assessment.severity)
    )
    breakdown_result = (await db.execute(breakdown_q)).all()
    severity_breakdown = [{"severity": r[0], "count": r[1]} for r in breakdown_result]

    return {
        "totals": {
            "patients": patients_count,
            "assessments": assessments_count,
            "highRiskCases": high_risk,
            "lowRiskCases": low_risk,
        },
        "severityBreakdown": severity_breakdown,
    }


@router.get("/alerts")
async def doctor_alerts(
    severity: Optional[List[str]] = Query(default=["Severe", "Moderately Severe"]),
    limit: int = Query(default=12, ge=1, le=50),
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Recent high-risk assessments."""
    result = await db.execute(
        select(Assessment)
        .where(Assessment.severity.in_(severity))
        .order_by(desc(Assessment.created_at))
        .limit(limit)
    )
    assessments = result.scalars().all()

    items = []
    for a in assessments:
        # Fetch the patient for this assessment
        patient_result = await db.execute(select(User).where(User.id == a.user_id))
        patient = patient_result.scalar_one_or_none()
        items.append({
            "assessmentId": a.id,
            "patient": {
                "id": patient.id if patient else a.user_id,
                "name": patient.name if patient else "Unknown",
                "email": patient.email if patient else "",
            },
            "severity": a.severity,
            "score": a.score_total,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
        })

    return {"items": items}


@router.get("/patient-trends")
async def patient_trends(
    patientId: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Assessment trend data per patient."""
    # Get patients with assessments
    if patientId:
        patients_result = await db.execute(
            select(User).where(User.id == patientId, User.role == "patient")
        )
        patients = patients_result.scalars().all()
    else:
        # Get patients who have assessments
        patient_ids_q = (
            select(Assessment.user_id)
            .group_by(Assessment.user_id)
            .order_by(desc(func.max(Assessment.created_at)))
            .limit(limit)
        )
        patient_ids = (await db.execute(patient_ids_q)).scalars().all()
        patients_result = await db.execute(
            select(User).where(User.id.in_(patient_ids))
        )
        patients = patients_result.scalars().all()

    result_patients = []
    for p in patients:
        assessments_result = await db.execute(
            select(Assessment)
            .where(Assessment.user_id == p.id)
            .order_by(Assessment.created_at)
        )
        assessments = assessments_result.scalars().all()

        points = []
        for i, a in enumerate(assessments):
            points.append({
                "session": f"S{i + 1}",
                "score": a.score_total,
                "severity": a.severity,
                "createdAt": a.created_at.isoformat() if a.created_at else None,
            })

        result_patients.append({
            "patient": {"id": p.id, "name": p.name},
            "points": points,
        })

    return {"patients": result_patients}
