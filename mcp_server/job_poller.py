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

# Config
VISUALIZER_API_URL = os.environ.get('VISUALIZER_API_URL', 'https://visualizer-tw4i.onrender.com')
OUTPUT_DIR = Path(os.environ.get('VISUALIZER_OUTPUT_DIR', '~/visualizer-results')).expanduser()
POLL_INTERVAL = 10  # seconds
NTFY_TOPIC = os.environ.get('VISUALIZER_NTFY_TOPIC', f'visualizer-{os.getenv("USER", "user")}')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def send_notification(title: str, message: str, sound: bool = True):
    """Send notification via native Linux desktop notification + sound."""
    # 1. Native Linux desktop notification (notify-send)
    try:
        subprocess.run(
            ['notify-send', '--urgency=critical', '--app-name=Visualizer', title, message],
            timeout=5,
            capture_output=True
        )
        print(f"📢 Notification sent: {title}")
    except Exception as e:
        print(f"⚠️  notify-send failed: {e}")

    # 2. Play sound
    if sound:
        # Try multiple sound options
        sound_files = [
            '/usr/share/sounds/freedesktop/stereo/complete.oga',
            '/usr/share/sounds/gnome/default/alerts/drip.ogg',
            '/usr/share/sounds/ubuntu/stereo/message.ogg',
            '/usr/share/sounds/sound-icons/trumpet-12.wav',
        ]

        sound_played = False
        for sound_file in sound_files:
            if Path(sound_file).exists():
                try:
                    # Try paplay first (PulseAudio)
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

        # Fallback: use speaker-test for a beep
        if not sound_played:
            try:
                subprocess.run(
                    ['speaker-test', '-t', 'sine', '-f', '1000', '-l', '1'],
                    timeout=2,
                    capture_output=True
                )
                print("🔊 Beep sound played")
            except:
                # Final fallback: terminal bell
                print("\a")  # ASCII bell
                print("🔔 Terminal bell attempted")

    # 3. Also send to ntfy.sh as backup (for mobile notifications)
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


def download_job_results(job_id: str, result: dict) -> list[str]:
    """Download all outputs for a completed job."""
    job_dir = OUTPUT_DIR / f"job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=True)

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
