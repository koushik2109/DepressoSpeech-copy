"""Assessment routes: create, list, latest, processing status, PHQ-8 questions."""

import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from database import get_db, async_session_factory
from src.models import User, Assessment, AssessmentAnswer, AssessmentMLDetail, MediaFile
from src.middleware.deps import get_current_user, require_patient
from src.services.ml_client import MLClient

logger = logging.getLogger("mindscope")

router = APIRouter(tags=["assessments"])

# ── PHQ-8 question data ───────────────────────────────

PHQ8_QUESTIONS = [
    {"id": 1, "text": "Over the last two weeks, how often have you felt little interest or pleasure in doing things?", "instruction": "Choose the option that best matches your experience."},
    {"id": 2, "text": "Over the last two weeks, how often have you felt down, depressed, or hopeless?", "instruction": "Choose the option that best matches your experience."},
    {"id": 3, "text": "Over the last two weeks, how often have you had trouble falling or staying asleep, or sleeping too much?", "instruction": "Choose the option that best matches your experience."},
    {"id": 4, "text": "Over the last two weeks, how often have you felt tired or had little energy?", "instruction": "Choose the option that best matches your experience."},
    {"id": 5, "text": "Over the last two weeks, how often have you had a poor appetite or been overeating?", "instruction": "Choose the option that best matches your experience."},
    {"id": 6, "text": "Over the last two weeks, how often have you felt bad about yourself, or that you are a failure?", "instruction": "Choose the option that best matches your experience."},
    {"id": 7, "text": "Over the last two weeks, how often have you had trouble concentrating on things, such as reading or watching television?", "instruction": "Choose the option that best matches your experience."},
    {"id": 8, "text": "Over the last two weeks, how often have you been moving or speaking so slowly that other people have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual?", "instruction": "Choose the option that best matches your experience."},
]

PHQ8_OPTIONS = [
    {"label": "Not at all", "value": 0},
    {"label": "Several days", "value": 1},
    {"label": "More than half the days", "value": 2},
    {"label": "Nearly every day", "value": 3},
]


def get_severity_label(score: int) -> str:
    if score <= 4:
        return "Minimal"
    if score <= 9:
        return "Mild"
    if score <= 14:
        return "Moderate"
    if score <= 19:
        return "Moderately Severe"
    return "Severe"


# ── Schemas ────────────────────────────────────────────

class AnswerInput(BaseModel):
    questionId: int = Field(..., ge=1, le=8)
    score: int = Field(..., ge=0, le=3)
    durationSec: Optional[float] = None
    audioFileId: Optional[str] = None


class CreateAssessmentRequest(BaseModel):
    questionSetVersion: str = "phq8_v1"
    answers: List[AnswerInput] = Field(..., min_length=1, max_length=8)
    recordingCount: int = Field(default=0, ge=0)


# ── GET /phq8/questions ────────────────────────────────

@router.get("/phq8/questions")
async def get_questions():
    # Questions never change — clients may cache for up to 1 hour
    content = {
        "version": "phq8_v1",
        "questions": PHQ8_QUESTIONS,
        "options": PHQ8_OPTIONS,
    }
    return JSONResponse(content=content, headers={"Cache-Control": "public, max-age=3600"})


# ── POST /assessments ─────────────────────────────────

@router.post("/assessments", status_code=201)
async def create_assessment(
    body: CreateAssessmentRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    # Compute total score
    score_total = sum(a.score for a in body.answers)
    severity = get_severity_label(score_total)

    assessment = Assessment(
        user_id=user.id,
        question_set_version=body.questionSetVersion,
        score_total=score_total,
        severity=severity,
        recording_count=body.recordingCount,
        status="processing" if body.recordingCount > 0 else "completed",
    )
    db.add(assessment)
    await db.flush()

    # Save individual answers
    audio_file_ids = []
    for ans in body.answers:
        answer = AssessmentAnswer(
            assessment_id=assessment.id,
            question_id=ans.questionId,
            score=ans.score,
            duration_sec=ans.durationSec,
            audio_file_id=ans.audioFileId,
        )
        db.add(answer)
        if ans.audioFileId:
            audio_file_ids.append(ans.audioFileId)
    await db.flush()

    assessment_id = assessment.id
    user_id = user.id

    # If audio was recorded, trigger background ML inference
    if audio_file_ids:
        background_tasks.add_task(_run_ml_inference, assessment_id, user_id, audio_file_ids)

    return {
        "assessment": {
            "id": assessment.id,
            "userId": assessment.user_id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
        }
    }


# ── GET /assessments/latest ───────────────────────────

@router.get("/assessments/latest")
async def get_latest_assessment(
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id == user.id)
        .order_by(desc(Assessment.created_at))
        .limit(1)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="No assessment found")

    # Build answers map
    answers_result = await db.execute(
        select(AssessmentAnswer).where(AssessmentAnswer.assessment_id == assessment.id)
    )
    answers_map = {str(a.question_id): a.score for a in answers_result.scalars().all()}

    return {
        "assessment": {
            "id": assessment.id,
            "score": assessment.score_total,
            "severity": assessment.severity,
            "answers": answers_map,
            "recordingCount": assessment.recording_count,
            "createdAt": assessment.created_at.isoformat() if assessment.created_at else None,
            "mlScore": assessment.ml_score,
            "mlSeverity": assessment.ml_severity,
        }
    }


# ── GET /assessments ──────────────────────────────────

@router.get("/assessments")
async def list_assessments(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    user: User = Depends(require_patient),
    db: AsyncSession = Depends(get_db),
):
    # Count
    count_q = select(func.count(Assessment.id)).where(Assessment.user_id == user.id)
    total = (await db.execute(count_q)).scalar() or 0

    # Fetch page
    offset = (page - 1) * pageSize
    result = await db.execute(
        select(Assessment)
        .where(Assessment.user_id == user.id)
        .order_by(desc(Assessment.created_at))
        .offset(offset)
        .limit(pageSize)
    )
    assessments = result.scalars().all()

    items = [
        {
            "id": a.id,
            "score": a.score_total,
            "severity": a.severity,
            "recordingCount": a.recording_count,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
            "mlScore": a.ml_score,
            "mlSeverity": a.ml_severity,
        }
        for a in assessments
    ]

    return {
        "items": items,
        "pagination": {
            "page": page,
            "pageSize": pageSize,
            "total": total,
        },
    }


# ── GET /assessments/{id}/processing-status ───────────

@router.get("/assessments/{assessment_id}/processing-status")
async def processing_status(
    assessment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if assessment.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(status_code=403, detail="Not authorized")

    # For now return completed status since scoring is synchronous
    return {
        "status": "completed" if assessment.status == "completed" else "processing",
        "progress": 100 if assessment.status == "completed" else 50,
        "stage": "Complete" if assessment.status == "completed" else "Analyzing responses",
    }


# ── Background ML inference task ─────────────────────

async def _run_ml_inference(assessment_id: str, user_id: str, audio_file_ids: list):
    """Background task: find audio files, send to ML model, store results."""
    import asyncio
    from pathlib import Path
    from config.settings import get_settings

    settings = get_settings()
    client = MLClient()

    try:
        async with async_session_factory() as db:
            # Find audio files
            result = await db.execute(
                select(MediaFile).where(MediaFile.id.in_(audio_file_ids))
            )
            media_files = result.scalars().all()

            if not media_files:
                logger.warning(f"[ML] No audio files found for assessment {assessment_id}")
                await db.execute(
                    select(Assessment).where(Assessment.id == assessment_id)
                )
                assessment = (await db.execute(
                    select(Assessment).where(Assessment.id == assessment_id)
                )).scalar_one_or_none()
                if assessment:
                    assessment.status = "completed"
                    await db.commit()
                return

            # Use the first audio file for ML prediction
            audio_file = media_files[0]
            audio_path = Path(settings.STORAGE_LOCAL_PATH) / audio_file.storage_key

            if not audio_path.exists():
                logger.error(f"[ML] Audio file not found on disk: {audio_path}")
                return

            # Call ML model
            ml_result = await client.predict_extended(
                audio_path=str(audio_path),
                participant_id=user_id,
            )

            # Update assessment with ML results
            assessment = (await db.execute(
                select(Assessment).where(Assessment.id == assessment_id)
            )).scalar_one_or_none()
            if assessment:
                assessment.ml_score = ml_result.get("phq8_score")
                assessment.ml_severity = ml_result.get("severity")
                assessment.ml_num_chunks = ml_result.get("num_chunks")
                assessment.status = "completed"

            # Store extended ML details
            confidence = ml_result.get("confidence", {})
            audio_quality = ml_result.get("audio_quality", {})
            detail = AssessmentMLDetail(
                assessment_id=assessment_id,
                confidence_mean=confidence.get("mean"),
                confidence_std=confidence.get("std"),
                ci_lower=confidence.get("ci_lower"),
                ci_upper=confidence.get("ci_upper"),
                audio_quality_score=audio_quality.get("quality"),
                audio_snr_db=audio_quality.get("snr_db"),
                audio_speech_prob=audio_quality.get("speech_prob"),
                behavioral_json=json.dumps(ml_result.get("behavioral", {})),
                inference_time_ms=(ml_result.get("inference_time_s", 0) * 1000),
            )
            db.add(detail)
            await db.commit()
            logger.info(f"[ML] Assessment {assessment_id}: score={ml_result.get('phq8_score')}")

    except Exception as e:
        logger.error(f"[ML] Inference failed for assessment {assessment_id}: {e}")
        try:
            async with async_session_factory() as db:
                assessment = (await db.execute(
                    select(Assessment).where(Assessment.id == assessment_id)
                )).scalar_one_or_none()
                if assessment:
                    assessment.status = "completed"
                    await db.commit()
        except Exception:
            pass


# ── GET /assessments/{id}/ml-details ─────────────────

@router.get("/assessments/{assessment_id}/ml-details")
async def get_ml_details(
    assessment_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    assessment = (await db.execute(
        select(Assessment).where(Assessment.id == assessment_id)
    )).scalar_one_or_none()

    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    if assessment.user_id != user.id and user.role not in ("admin", "doctor"):
        raise HTTPException(status_code=403, detail="Not authorized")

    detail = (await db.execute(
        select(AssessmentMLDetail).where(AssessmentMLDetail.assessment_id == assessment_id)
    )).scalar_one_or_none()

    if not detail:
        return {"mlDetails": None}

    return {
        "mlDetails": {
            "confidenceMean": detail.confidence_mean,
            "confidenceStd": detail.confidence_std,
            "ciLower": detail.ci_lower,
            "ciUpper": detail.ci_upper,
            "audioQualityScore": detail.audio_quality_score,
            "audioSnrDb": detail.audio_snr_db,
            "audioSpeechProb": detail.audio_speech_prob,
            "behavioral": json.loads(detail.behavioral_json) if detail.behavioral_json else {},
            "inferenceTimeMs": detail.inference_time_ms,
        }
    }
