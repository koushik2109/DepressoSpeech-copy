"""Email service for sending OTP verification emails.

Fixed: Uses asyncio.to_thread() for non-blocking SMTP operations.
"""

import asyncio
import logging
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config.settings import get_settings

logger = logging.getLogger("mindscope.email")
settings = get_settings()


def generate_otp(length: int = 6) -> str:
    """Generate a random numeric OTP."""
    return "".join(random.choices(string.digits, k=length))


def _send_smtp(to_email: str, msg: MIMEMultipart) -> bool:
    """Synchronous SMTP send — called via asyncio.to_thread()."""
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        logger.info(f"[EMAIL] OTP email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send OTP to {to_email}: {e}")
        return False


def send_otp_email(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send an OTP verification email (synchronous version for backward compat).

    Args:
        to_email: Recipient email address
        otp: The OTP code to send
        user_name: User's display name

    Returns:
        True if email was sent successfully, False otherwise
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "[EMAIL] SMTP not configured — OTP email skipped. "
            "Set SMTP_USER and SMTP_PASSWORD in .env"
        )
        logger.info(f"[EMAIL-DEV] OTP for {to_email}: {otp}")
        return True

    msg = _build_otp_message(to_email, otp, user_name)
    return _send_smtp(to_email, msg)


async def send_otp_email_async(to_email: str, otp: str, user_name: str = "User") -> bool:
    """Send an OTP verification email without blocking the event loop.

    Uses asyncio.to_thread() to run SMTP in a thread pool.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning(
            "[EMAIL] SMTP not configured — OTP email skipped. "
            "Set SMTP_USER and SMTP_PASSWORD in .env"
        )
        logger.info(f"[EMAIL-DEV] OTP for {to_email}: {otp}")
        return True

    msg = _build_otp_message(to_email, otp, user_name)
    return await asyncio.to_thread(_send_smtp, to_email, msg)


def _build_otp_message(to_email: str, otp: str, user_name: str) -> MIMEMultipart:
    """Build the OTP email message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"MindScope — Your Verification Code: {otp}"
    msg["From"] = f"MindScope <{settings.SMTP_USER}>"
    msg["To"] = to_email

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f4fcf7; margin: 0; padding: 40px 20px; }}
            .container {{ max-width: 480px; margin: 0 auto; background: #ffffff; border-radius: 16px; border: 1px solid #d8f3dc; overflow: hidden; box-shadow: 0 4px 24px rgba(45, 106, 79, 0.08); }}
            .header {{ background: linear-gradient(135deg, #1b3a2d, #2d6a4f); padding: 32px 32px 28px; text-align: center; }}
            .header h1 {{ color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 0.5px; }}
            .header p {{ color: #b7e4c7; margin: 8px 0 0; font-size: 13px; }}
            .body {{ padding: 32px; text-align: center; }}
            .greeting {{ color: #1b1b1b; font-size: 18px; margin: 0 0 12px; }}
            .message {{ color: #666; font-size: 14px; line-height: 1.6; margin: 0 0 28px; }}
            .otp-box {{ display: inline-block; background: linear-gradient(135deg, #f4fcf7, #d8f3dc); border: 2px solid #2d6a4f; border-radius: 12px; padding: 16px 40px; margin: 0 0 24px; }}
            .otp-code {{ font-size: 36px; font-weight: 700; color: #1b3a2d; letter-spacing: 8px; font-family: 'Courier New', monospace; }}
            .expire {{ color: #999; font-size: 12px; margin: 0; }}
            .footer {{ background: #fafffe; padding: 20px 32px; border-top: 1px solid #e8e8e8; text-align: center; }}
            .footer p {{ color: #999; font-size: 11px; margin: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>MindScope</h1>
                <p>Email Verification</p>
            </div>
            <div class="body">
                <p class="greeting">Hi {user_name},</p>
                <p class="message">Use the verification code below to complete your account setup. This code will expire in {settings.OTP_EXPIRE_MINUTES} minutes.</p>
                <div class="otp-box">
                    <span class="otp-code">{otp}</span>
                </div>
                <p class="expire">Expires in {settings.OTP_EXPIRE_MINUTES} minutes</p>
            </div>
            <div class="footer">
                <p>If you did not request this code, please ignore this email.</p>
            </div>
        </div>
    </body>
    </html>
    """

    text_body = (
        f"Hi {user_name},\n\n"
        f"Your MindScope verification code is: {otp}\n\n"
        f"This code expires in {settings.OTP_EXPIRE_MINUTES} minutes.\n\n"
        f"If you did not request this, ignore this email."
    )

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg
