"""
Email Fetcher
-------------
IMAP-based Gmail retriever with support for:
  - Full inbox fetch
  - Unread-only fetch
  - Real-time monitoring (polling loop)
"""

import imaplib
import email
import time
import logging
from email.header import decode_header
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class EmailFetcher:
    """
    Parameters
    ----------
    email_address : str
        Gmail address, e.g. ``user@gmail.com``.
    app_password : str
        Google App Password (16-char, generated in Google Account settings).
    imap_server : str
        Defaults to ``imap.gmail.com``.
    """

    def __init__(self, email_address: str, app_password: str,
                 imap_server: str = "imap.gmail.com"):
        self.email_address = email_address
        self.app_password = app_password
        self.imap_server = imap_server
        self.connection: Optional[imaplib.IMAP4_SSL] = None

    # ── Connection lifecycle ──────────────────────────────────────────────

    def connect(self) -> bool:
        try:
            self.connection = imaplib.IMAP4_SSL(self.imap_server)
            self.connection.login(self.email_address, self.app_password)
            logger.info("Connected to %s", self.email_address)
            print(f"[Fetcher] Connected → {self.email_address}")
            return True
        except Exception as exc:
            logger.error("Connection failed: %s", exc)
            print(f"[Fetcher] Connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
                print("[Fetcher] Disconnected.")
            except Exception as exc:
                logger.warning("Disconnect error: %s", exc)
            finally:
                self.connection = None

    # ── Fetch helpers ─────────────────────────────────────────────────────

    def get_inbox_emails(self, limit: int = 10) -> List[dict]:
        return self._search_and_fetch("ALL", limit)

    def get_unread_emails(self, limit: int = 10) -> List[dict]:
        return self._search_and_fetch("UNSEEN", limit)

    def _search_and_fetch(self, criterion: str, limit: int) -> List[dict]:
        if not self.connection:
            logger.error("Not connected. Call connect() first.")
            return []
        try:
            self.connection.select("INBOX")
            _, messages = self.connection.search(None, criterion)
            ids = messages[0].split()
            emails = []
            for eid in ids[-limit:]:
                data = self._fetch_email(eid)
                if data:
                    emails.append(data)
            return emails
        except Exception as exc:
            logger.error("Fetch error (%s): %s", criterion, exc)
            return []

    def _fetch_email(self, email_id) -> Optional[dict]:
        try:
            _, msg_data = self.connection.fetch(email_id, "(RFC822)")
            for part in msg_data:
                if isinstance(part, tuple):
                    msg = email.message_from_bytes(part[1])
                    return {
                        "id":      email_id.decode() if isinstance(email_id, bytes) else str(email_id),
                        "subject": self._decode_subject(msg["Subject"]),
                        "from":    msg["From"] or "Unknown",
                        "date":    msg["Date"] or "",
                        "body":    self._get_body(msg),
                    }
        except Exception as exc:
            logger.warning("Could not fetch email %s: %s", email_id, exc)
        return None

    @staticmethod
    def _decode_subject(raw) -> str:
        if not raw:
            return "No Subject"
        decoded_parts = decode_header(raw)
        result = ""
        for chunk, enc in decoded_parts:
            if isinstance(chunk, bytes):
                result += chunk.decode(enc or "utf-8", errors="ignore")
            else:
                result += chunk
        return result.strip()

    @staticmethod
    def _get_body(msg) -> str:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    except Exception:
                        body = ""
                    break
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            except Exception:
                body = ""
        return body.strip()

    # ── Real-time monitoring ──────────────────────────────────────────────

    def monitor_inbox(
        self,
        callback: Callable[[dict], None],
        interval: int = 60,
    ) -> None:
        """
        Poll the inbox every *interval* seconds and call *callback* for each
        newly seen email.  Runs until KeyboardInterrupt.
        """
        print(f"\n{'='*60}")
        print("REAL-TIME EMAIL MONITORING STARTED")
        print(f"Polling every {interval}s — Ctrl+C to stop")
        print(f"{'='*60}\n")

        seen_ids: set = set()

        while True:
            try:
                for email_data in self.get_unread_emails(limit=10):
                    eid = email_data["id"]
                    if eid not in seen_ids:
                        print(f"[Monitor] New email: {email_data['subject'][:60]}")
                        callback(email_data)
                        seen_ids.add(eid)
                time.sleep(interval)
            except KeyboardInterrupt:
                print("\n[Monitor] Stopped by user.")
                break
            except Exception as exc:
                logger.error("Monitor error: %s", exc)
                time.sleep(interval)
