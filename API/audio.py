"""
Download the audio for every video in video_data.csv with yt-dlp, in parallel,
into an `audio/` folder next to this script.
 
Setup (Mac):
    uv venv --python 3.12 .venv      # yt-dlp needs Python 3.11+; system python3 is 3.9
    uv pip install --python .venv/bin/python -U "yt-dlp[default]" pandas tqdm   # [default] includes yt-dlp-ejs
    brew install deno ffmpeg      # Deno: needed for YouTube. ffmpeg: needed for CLIP_SECONDS
 
Run:
    .venv/bin/python audio.py

If every video fails with "The page needs to be reloaded", update yt-dlp first (re-run the uv pip line).
 
Output:
    audio/<first 2 chars of ID>/<video ID>.<ext>   the audio files (.webm Opus or .m4a)
    audio_log.jsonl                                one line per video: path or error
    audio_index.csv                                Video ID -> audio_path / audio_error
 
Resumable: re-running skips every video already downloaded or permanently
unavailable, and retries the rest.
"""
 
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
 
import pandas as pd
import yt_dlp
from tqdm import tqdm
from yt_dlp.utils import DownloadError, download_range_func
 
# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
INPUT = "video_data1.csv"
ID_COL = "Video ID"
AUDIO_DIR = "audio"
LOG_PATH = "audio_log.jsonl"
INDEX_PATH = "audio_index.csv"
 
MAX_WORKERS = 1            # one worker stays under the ~300 videos/hour limit without an account
SLEEP_RANGE = (5.0, 10.0)  # wait between videos (random); yt-dlp recommends 5-10 s
CLIP_SECONDS = None        # e.g. 60 = keep only the first 60 s (needs ffmpeg); None = full audio
LOW_BITRATE = True         # smallest Opus stream (~50 kbps, fine for speech models)
MIN_FREE_GB = 5            # stop before the disk fills up
BLOCK_COOLDOWN = 3600      # seconds to pause when YouTube blocks us; these blocks last well over 30 min
MAX_BLOCKS_IN_A_ROW = 5    # give up after this many blocks with no success between
PROXY = None               # e.g. "http://user:pass@host:port" for a rotating proxy
COOKIES_FILE = None        # e.g. "cookies.txt" exported from a throwaway account; never commit this file
 
# yt-dlp error text -> our label. Permanent ones are never retried.
PERMANENT_PATTERNS = {
    "Private video": "PrivateVideo",
    "Video unavailable": "VideoUnavailable",
    "This video has been removed": "VideoRemoved",
    "account associated with this video has been terminated": "AccountTerminated",
    "members-only": "MembersOnly",
    "Join this channel": "MembersOnly",
    "Sign in to confirm your age": "AgeRestricted",
    "copyright": "CopyrightTakedown",
}
BLOCK_PATTERNS = ["not a bot", "HTTP Error 429", "Too Many Requests", "rate-limited"]
 
 
# ----------------------------------------------------------------------------
# Shared state between workers
# ----------------------------------------------------------------------------
_thread_local = threading.local()
_log_lock = threading.Lock()
_resume_at = 0.0                  # time.time() before which nobody downloads
_resume_lock = threading.Lock()
_blocks_in_a_row = 0
_stop = threading.Event()
 
 
def ydl_options():
    fmt = "wa[acodec^=opus]/wa" if LOW_BITRATE else "ba[acodec^=opus]/ba"
    opts = {
        "format": fmt,                          # audio-only stream, never video
        "outtmpl": os.path.join(AUDIO_DIR, "%(id).2s", "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "sleep_interval_requests": 0.75,        # yt-dlp's own suggested pacing
        "overwrites": False,
    }
    if CLIP_SECONDS:
        opts["download_ranges"] = download_range_func(None, [(0, CLIP_SECONDS)])
    if COOKIES_FILE:
        opts["cookiefile"] = COOKIES_FILE
    if PROXY:
        opts["proxy"] = PROXY
    return opts
 
 
def get_ydl():
    """One YoutubeDL instance per thread (instances aren't safe to share)."""
    if not hasattr(_thread_local, "ydl"):
        _thread_local.ydl = yt_dlp.YoutubeDL(ydl_options())
    return _thread_local.ydl
 
 
def classify(message):
    """Return (label, kind) where kind is 'permanent', 'blocked' or 'retry'."""
    for pattern in BLOCK_PATTERNS:
        if pattern.lower() in message.lower():
            return "Blocked", "blocked"
    for pattern, label in PERMANENT_PATTERNS.items():
        if pattern.lower() in message.lower():
            return label, "permanent"
    first_line = message.strip().splitlines()[0] if message.strip() else "Unknown"
    return first_line.replace("ERROR: ", "")[:200], "retry"
 
 
def wait_if_paused():
    while not _stop.is_set():
        remaining = _resume_at - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5))
 
 
def download_one(video_id):
    """Download one video's audio. Returns a log record (dict)."""
    global _resume_at, _blocks_in_a_row
    wait_if_paused()
    if _stop.is_set():
        return None
 
    url = f"https://www.youtube.com/watch?v={video_id}"
    record = {"video_id": video_id, "path": None, "error": None, "kind": None}
    try:
        info = get_ydl().extract_info(url, download=True)
        downloads = info.get("requested_downloads") or [{}]
        record["path"] = downloads[0].get("filepath")
        record["ext"] = info.get("ext")
        record["abr"] = info.get("abr")
        record["duration"] = info.get("duration")
        with _resume_lock:
            _blocks_in_a_row = 0
    except DownloadError as e:
        label, kind = classify(str(e))
        record["error"], record["kind"] = label, kind
        if kind == "blocked":
            with _resume_lock:
                _blocks_in_a_row += 1
                _resume_at = max(_resume_at, time.time() + BLOCK_COOLDOWN)
                if _blocks_in_a_row >= MAX_BLOCKS_IN_A_ROW:
                    _stop.set()
            tqdm.write(f"[{video_id}] blocked by YouTube; all workers pausing "
                       f"{BLOCK_COOLDOWN}s ({_blocks_in_a_row}/{MAX_BLOCKS_IN_A_ROW})")
    except Exception as e:  # network errors, disk errors, anything unexpected
        record["error"], record["kind"] = f"{type(e).__name__}: {str(e)[:200]}", "retry"
 
    time.sleep(random.uniform(*SLEEP_RANGE))  # be polite between videos
    return record
 
 
def load_log():
    """Latest record per video from previous runs."""
    log = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written line from a sudden stop
                log[rec["video_id"]] = rec
    return log
 
 
def is_done(rec):
    if rec.get("path") and os.path.exists(rec["path"]):
        return True
    return rec.get("kind") == "permanent"
 
 
def retry_priority(rec):
    """0 = blocked last time, 1 = failed some other way, 2 = never tried."""
    if rec is None:
        return 2
    return 0 if rec.get("kind") == "blocked" else 1


def read_video_ids(path=INPUT):
    """Read only the ID column, in chunks, so the 2.8 GB file never sits in memory."""
    ids = []
    for chunk in pd.read_csv(path, usecols=[ID_COL], chunksize=200_000):
        ids.extend(chunk[ID_COL].dropna().astype(str))
    return list(dict.fromkeys(ids))  # unique, original order
 
 
def download_audio(video_ids, max_workers=MAX_WORKERS):
    """Download audio for every ID not already done. Returns the full log dict."""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    log = load_log()
    todo = [vid for vid in video_ids if vid not in log or not is_done(log[vid])]
    todo.sort(key=lambda vid: retry_priority(log.get(vid)))  # stable: keeps CSV order within each group
    print(f"{len(video_ids):,} videos, {len(video_ids) - len(todo):,} already done, "
          f"{len(todo):,} to download")
    if not todo:
        return log
 
    ok = failed = 0
    with open(LOG_PATH, "a", encoding="utf-8") as log_file, \
         ThreadPoolExecutor(max_workers=max_workers) as pool, \
         tqdm(total=len(todo), desc="Audio", unit="video") as pbar:
 
        # Submit gradually so a stop doesn't leave thousands of queued jobs
        pending, queue = set(), iter(todo)
        for _ in range(max_workers * 2):
            vid = next(queue, None)
            if vid:
                pending.add(pool.submit(download_one, vid))
 
        while pending:
            future = next(as_completed(pending))
            pending.remove(future)
            rec = future.result()
            if rec is not None:
                log[rec["video_id"]] = rec
                with _log_lock:
                    log_file.write(json.dumps(rec) + "\n")
                    log_file.flush()
                if rec["path"]:
                    ok += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, failed=failed)
 
            free_gb = shutil.disk_usage(AUDIO_DIR).free / 1e9
            if free_gb < MIN_FREE_GB and not _stop.is_set():
                tqdm.write(f"Only {free_gb:.1f} GB free; stopping.")
                _stop.set()
 
            if not _stop.is_set():
                vid = next(queue, None)
                if vid:
                    pending.add(pool.submit(download_one, vid))
 
    if _stop.is_set():
        print("Stopped early. Re-run later (or with a proxy) to continue where it left off.")
    return log
 
 
def write_index(video_ids, log):
    rows = [{ID_COL: vid,
             "audio_path": log.get(vid, {}).get("path"),
             "audio_error": log.get(vid, {}).get("error")} for vid in video_ids]
    pd.DataFrame(rows).to_csv(INDEX_PATH, index=False)
    print(f"Wrote {INDEX_PATH}")
 
 
if __name__ == "__main__":
    video_ids = read_video_ids()
    # video_ids = video_ids[:50]   # uncomment to test on a small sample first
    log = download_audio(video_ids)
    write_index(video_ids, log)