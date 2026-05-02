"""Doctor dashboard routes: summary, alerts, patient trends."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from database import get_db
from src.models import User, Assessment, Doctor, DoctorAssignment
from src.middleware.deps import require_doctor

router = APIRouter(prefix="/doctor/dashboard", tags=["doctor"])


async def _doctor_context(user: User, db: AsyncSession) -> Doctor:
    result = await db.execute(select(Doctor).where(Doctor.user_id == user.id))
    doctor = result.scalar_one_or_none()
    if not doctor:
        doctor = Doctor(
            user_id=user.id,
            name=user.name,
            email=user.email,
            fee=100.0,
            is_available=False,
        )
        db.add(doctor)
        await db.flush()
    return doctor


async def _doctor_assignment_rows(doctor_id: str, db: AsyncSession):
    result = await db.execute(
        select(DoctorAssignment)
        .where(DoctorAssignment.doctor_id == doctor_id)
        .order_by(desc(DoctorAssignment.created_at))
    )
    return result.scalars().all()


@router.get("/summary")
async def doctor_summary(
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate stats for the doctor dashboard."""
    doctor = await _doctor_context(user, db)
    assignments = await _doctor_assignment_rows(doctor.id, db)
    patient_ids = list({item.patient_id for item in assignments if item.patient_id})
    assessments = []
    if patient_ids:
        result = await db.execute(
            select(Assessment)
            .where(Assessment.user_id.in_(patient_ids))
            .order_by(desc(Assessment.created_at))
        )
        assessments = result.scalars().all()

    active_patient_count = (await db.execute(
        select(func.count(func.distinct(DoctorAssignment.patient_id))).where(
            DoctorAssignment.doctor_id == doctor.id,
            DoctorAssignment.status.in_(["accepted", "completed"]),
        )
    )).scalar() or 0
    if doctor.patient_count != active_patient_count:
        doctor.patient_count = active_patient_count
        await db.flush()
    patient_count = active_patient_count
    assessments_count = len(assessments)
    high_risk = sum(1 for item in assessments if item.severity in ["Severe", "Moderately Severe"])
    low_risk = sum(1 for item in assessments if item.severity in ["Minimal", "Mild"])
    severity_breakdown_map = {}
    for item in assessments:
        severity_breakdown_map[item.severity] = severity_breakdown_map.get(item.severity, 0) + 1
    severity_breakdown = [
        {"severity": severity, "count": count}
        for severity, count in severity_breakdown_map.items()
    ]

    return {
        "patientCount": patient_count,
        "totals": {
            "patients": patient_count,
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
    doctor = await _doctor_context(user, db)
    assignments = await _doctor_assignment_rows(doctor.id, db)
    patient_ids = list({item.patient_id for item in assignments if item.patient_id})
    if not patient_ids:
        return {"items": []}

    result = await db.execute(
        select(Assessment)
        .where(
            Assessment.user_id.in_(patient_ids),
            Assessment.severity.in_(severity),
        )
        .order_by(desc(Assessment.created_at))
        .limit(limit)
    )
    assessments = result.scalars().all()

    user_ids = list({a.user_id for a in assessments})
    if user_ids:
        patients_result = await db.execute(select(User).where(User.id.in_(user_ids)))
        patients_map = {p.id: p for p in patients_result.scalars().all()}
    else:
        patients_map = {}

    items = []
    for a in assessments:
        patient = patients_map.get(a.user_id)
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
    user: User = Depends(require_doctor),
    db: AsyncSession = Depends(get_db),
):
    """Assessment trend data per patient."""
    doctor = await _doctor_context(user, db)
    assignments = await _doctor_assignment_rows(doctor.id, db)
    assigned_patient_ids = list({assignment.patient_id for assignment in assignments if assignment.patient_id})
    patient_ids = [patientId] if patientId else assigned_patient_ids
    patient_ids = [pid for pid in patient_ids if pid in assigned_patient_ids]
    if not patient_ids:
        return {"patients": []}

    patients_result = await db.execute(
        select(User).where(User.id.in_(patient_ids), User.role == "patient")
    )
    patients = patients_result.scalars().all()

    assessments_result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id.in_(patient_ids))
        .order_by(Assessment.user_id, Assessment.created_at)
    )
    all_assessments = assessments_result.scalars().all()
    assessments_by_patient: dict[str, list[Assessment]] = {}
    for assessment in all_assessments:
        assessments_by_patient.setdefault(assessment.user_id, []).append(assessment)

    result_patients = []
    for p in patients:
        points = []
        for i, a in enumerate(assessments_by_patient.get(p.id, [])):
            points.append({
                "sessionId": a.id,
                "session": f"S{i + 1}",
                "timestamp": a.created_at.isoformat() if a.created_at else None,
                "createdAt": a.created_at.isoformat() if a.created_at else None,
                "score": a.score_total,
                "severity": a.severity,
            })

        result_patients.append({
            "patient": {"id": p.id, "name": p.name},
            "points": points,
        })

    return {"patients": result_patients}
