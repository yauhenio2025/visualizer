#!/usr/bin/env python3 -u
"""
Standalone job poller - monitors visualizer jobs and downloads results automatically.
Runs independently of Claude Code/MCP.

Usage:
    python job_poller.py job_id1 job_id2 job_id3 ...

Or pipe from stdin:
    echo "job_id1 job_id2" | python job_poller.py
"""

import os
import sys
import platform
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Detect OS
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'
IS_WINDOWS = platform.system() == 'Windows'

# Config
VISUALIZER_API_URL = os.environ.get('VISUALIZER_API_URL', 'https://visualizer-tw4i.onrender.com')
OUTPUT_DIR = Path(os.environ.get('VISUALIZER_OUTPUT_DIR', '~/visualizer-results')).expanduser()
POLL_INTERVAL = 10  # seconds
NTFY_TOPIC = os.environ.get('VISUALIZER_NTFY_TOPIC', f'visualizer-{os.getenv("USER", "user")}')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def send_notification(title: str, message: str, sound: bool = True):
    """Send notification via native desktop notification + sound (cross-platform)."""
    notification_sent = False
    sound_played = False

    # === MACOS ===
    if IS_MACOS:
        # 1. macOS notification via osascript
        try:
            script = f'display notification "{message}" with title "{title}" sound name "Glass"'
            subprocess.run(
                ['osascript', '-e', script],
                timeout=5,
                capture_output=True
            )
            print(f"📢 Notification sent: {title}")
            notification_sent = True
            if sound:
                sound_played = True  # Sound included in notification
        except Exception as e:
            print(f"⚠️  osascript notification failed: {e}")

        # 2. macOS sound fallback via afplay
        if sound and not sound_played:
            macos_sounds = [
                '/System/Library/Sounds/Glass.aiff',
                '/System/Library/Sounds/Ping.aiff',
                '/System/Library/Sounds/Pop.aiff',
                '/System/Library/Sounds/Purr.aiff',
            ]
            for sound_file in macos_sounds:
                if Path(sound_file).exists():
                    try:
                        result = subprocess.run(
                            ['afplay', sound_file],
                            timeout=5,
                            capture_output=True
                        )
                        if result.returncode == 0:
                            print(f"🔊 Sound played: {sound_file}")
                            sound_played = True
                            break
                    except:
                        pass

    # === LINUX ===
    elif IS_LINUX:
        # 1. Linux notification via notify-send
        try:
            subprocess.run(
                ['notify-send', '--urgency=critical', '--app-name=Visualizer', title, message],
                timeout=5,
                capture_output=True
            )
            print(f"📢 Notification sent: {title}")
            notification_sent = True
        except Exception as e:
            print(f"⚠️  notify-send failed: {e}")

        # 2. Linux sound via paplay
        if sound:
            linux_sounds = [
                '/usr/share/sounds/freedesktop/stereo/complete.oga',
                '/usr/share/sounds/freedesktop/stereo/bell.oga',
                '/usr/share/sounds/gnome/default/alerts/drip.ogg',
                '/usr/share/sounds/ubuntu/stereo/message.ogg',
            ]
            for sound_file in linux_sounds:
                if Path(sound_file).exists():
                    try:
                        result = subprocess.run(
                            ['paplay', sound_file],
                            timeout=5,
                            capture_output=True
                        )
                        if result.returncode == 0:
                            print(f"🔊 Sound played: {sound_file}")
                            sound_played = True
                            break
                    except:
                        pass

            # Fallback: speaker-test beep
            if not sound_played:
                try:
                    subprocess.run(
                        ['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1'],
                        timeout=2,
                        capture_output=True
                    )
                    print("🔊 Beep sound played")
                    sound_played = True
                except:
                    pass

    # === WINDOWS ===
    elif IS_WINDOWS:
        # Windows notification via PowerShell
        try:
            ps_script = f'''
            [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
            $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
            $template.SelectSingleNode("//text[@id='1']").InnerText = "{title}"
            $template.SelectSingleNode("//text[@id='2']").InnerText = "{message}"
            [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Visualizer").Show($template)
            '''
            subprocess.run(['powershell', '-Command', ps_script], timeout=10, capture_output=True)
            print(f"📢 Notification sent: {title}")
            notification_sent = True
        except:
            pass

        # Windows sound
        if sound:
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                print("🔊 Windows sound played")
                sound_played = True
            except:
                pass

    # === FALLBACK: Terminal bell ===
    if sound and not sound_played:
        print("\a")  # ASCII bell
        print("🔔 Terminal bell attempted")

    # === NTFY.SH for mobile/remote notifications ===
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Priority": "4" if sound else "1",
                "Tags": "visualizer"
            },
            timeout=5
        )
    except:
        pass


def check_job_status(job_id: str) -> dict:
    """Check job status from API."""
    try:
        response = requests.get(
            f"{VISUALIZER_API_URL}/api/analyzer/jobs/{job_id}",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "status": "error"}


def get_job_result(job_id: str) -> dict:
    """Get full job result with output URLs."""
    try:
        response = requests.get(
            f"{VISUALIZER_API_URL}/api/analyzer/jobs/{job_id}/result",
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def download_file(url: str, output_path: Path) -> tuple[bool, str]:
    """Download file from URL."""
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, ""
    except Exception as e:
        return False, str(e)


def generate_meaningful_folder_name(result: dict, job_id: str) -> str:
    """Generate a meaningful folder name from job result data.

    Format: YYYYMMDD_HHMMSS_engine_DocumentTitle
    Example: 20251218_200113_dialectical_structure_Four_Forms_Critical_Theory
    """
    import re

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Extract engine name(s) from outputs or extended_info
    engine_parts = []
    outputs = result.get("outputs", {})
    if outputs:
        engine_parts = list(outputs.keys())[:2]

    if not engine_parts:
        engine_parts = [result.get("extended_info", {}).get("engine", "analysis")]

    # Extract document title from extended_info
    doc_title = ""
    extended_info = result.get("extended_info", {})
    documents = extended_info.get("documents", [])

    if documents:
        first_doc = documents[0]
        doc_title = first_doc.get("title", "") or first_doc.get("id", "")

        if doc_title:
            doc_title = re.sub(r'\.(pdf|txt|md|docx)$', '', doc_title, flags=re.IGNORECASE)
            doc_title = re.sub(r'^[A-Za-z]+\s*-\s*\d{4}\s*-\s*', '', doc_title)
            doc_title = re.sub(r'[^a-zA-Z0-9\s]', '', doc_title)
            doc_title = re.sub(r'\s+', '_', doc_title.strip())
            if len(doc_title) > 50:
                doc_title = doc_title[:47] + "..."

    parts = [timestamp]
    if engine_parts:
        parts.append("_".join(engine_parts[:2]))
    if doc_title:
        parts.append(doc_title)
    else:
        parts.append(job_id[:8])

    folder_name = "_".join(parts)
    folder_name = re.sub(r'[<>:"/\\|?*]', '', folder_name)
    folder_name = re.sub(r'_+', '_', folder_name)
    folder_name = folder_name.strip('_')

    return folder_name


def download_job_results(job_id: str, result: dict) -> list[str]:
    """Download all outputs for a completed job."""
    folder_name = generate_meaningful_folder_name(result, job_id)
    job_dir = OUTPUT_DIR / folder_name
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"  📁 Folder: {folder_name}")

    downloaded = []
    outputs = result.get("outputs", {})

    for engine_key, engine_result in outputs.items():
        if not isinstance(engine_result, dict):
            continue

        # Check for image_url
        image_url = engine_result.get("image_url")
        if image_url:
            ext = ".png" if ".png" in image_url.lower() else ".jpg"
            file_path = job_dir / f"{engine_key}{ext}"
            success, error = download_file(image_url, file_path)
            if success:
                downloaded.append(str(file_path))
                print(f"  ✓ Downloaded: {engine_key}{ext}")
            else:
                print(f"  ✗ Failed {engine_key}: {error}")
            continue

        # Check for text content
        text = engine_result.get("text") or engine_result.get("content")
        if text and isinstance(text, str):
            file_path = job_dir / f"{engine_key}.md"
            file_path.write_text(text, encoding='utf-8')
            downloaded.append(str(file_path))
            print(f"  ✓ Saved: {engine_key}.md")

    # Save raw JSON
    json_path = job_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    downloaded.append(str(json_path))

    return downloaded


def monitor_job(job_id: str) -> dict:
    """Monitor a single job until completion."""
    print(f"\n[{job_id[:8]}...] Monitoring...")

    while True:
        status_data = check_job_status(job_id)
        status = status_data.get("status", "unknown")

        if status == "completed":
            print(f"[{job_id[:8]}...] ✓ Completed!")
            result = get_job_result(job_id)
            downloaded = download_job_results(job_id, result)
            return {"job_id": job_id, "status": "completed", "files": downloaded}

        elif status == "failed":
            error = status_data.get("error_message", "Unknown error")
            print(f"[{job_id[:8]}...] ✗ Failed: {error}")
            return {"job_id": job_id, "status": "failed", "error": error}

        elif status == "error":
            print(f"[{job_id[:8]}...] ✗ API Error: {status_data.get('error')}")
            return {"job_id": job_id, "status": "error", "error": status_data.get("error")}

        else:
            # Still processing
            print(f"[{job_id[:8]}...] Status: {status}", end="\r")
            time.sleep(POLL_INTERVAL)


def main():
    # Get job IDs from args or stdin
    if len(sys.argv) > 1:
        job_ids = sys.argv[1:]
    else:
        # Read from stdin
        input_text = sys.stdin.read().strip()
        job_ids = input_text.split()

    if not job_ids:
        print("Usage: python job_poller.py job_id1 job_id2 ...")
        sys.exit(1)

    print(f"Monitoring {len(job_ids)} job(s)...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("-" * 50)

    results = []

    # Monitor all jobs (could parallelize, but sequential is clearer)
    for job_id in job_ids:
        result = monitor_job(job_id)
        results.append(result)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]

    print(f"Completed: {len(completed)}/{len(results)}")

    total_files = sum(len(r.get("files", [])) for r in completed)
    print(f"Files downloaded: {total_files}")

    if failed:
        print(f"\nFailed jobs:")
        for r in failed:
            print(f"  - {r['job_id'][:8]}...: {r.get('error', r['status'])}")

    # Send notification
    send_notification(
        "✅ Jobs Complete",
        f"Downloaded {total_files} files from {len(completed)} jobs",
        sound=True
    )

    # Open output directory
    if completed:
        print(f"\nResults in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
