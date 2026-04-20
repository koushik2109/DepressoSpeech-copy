from .auth import router as auth_router
from .assessments import router as assessments_router
from .audio import router as audio_router
from .doctor import router as doctor_router
from .admin import router as admin_router

__all__ = [
    "auth_router",
    "assessments_router",
    "audio_router",
    "doctor_router",
    "admin_router",
]
