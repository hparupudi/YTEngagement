"""
Fetch the original-language ("default") YouTube transcript for every video in a
CSV file and add these columns:

    transcript               full transcript text (None if it couldn't be fetched)
    transcript_language      language code of that transcript, which doubles as
                             the video's spoken language
    transcript_is_generated  True if it's YouTube's auto-generated captions,
                             False if it was uploaded manually
    transcript_error         why there's no transcript, e.g. "TranscriptsDisabled",
                             "VideoUnavailable", "RequestBlocked" (None on success)

No language list is needed: each video's available tracks are listed and the
original-language one is picked automatically.

Accepts files from collect_videos.py, including splits of them. Video IDs come from
a `video_id` or `Video ID` column, or from the `url` column if neither exists.

Run:
    .venv/bin/python transcripts.py video_data1.csv

Avoiding blocks:
    The script waits a random 5-10 seconds between videos. If YouTube blocks the
    IP, it pauses for an hour and tries again, and stops cleanly after 5 blocks
    in a row (progress is kept). To use a proxy, set TRANSCRIPT_PROXY in .env or
    pass --proxy, e.g. http://user:pass@host:port.

Language:
    When the CSV has default_audio_language or detected_language (collect_videos.py
    output does), the transcript in that language is preferred. This matters for
    channels that add dubbed audio tracks, which each get their own captions.

Output, named after the input file:
    video_data1_with_transcripts.csv     the input columns plus the new ones
    video_data1_transcript_cache.jsonl   every result so far

Re-running the same command resumes from the cache. Each input file gets its own
cache, so splits can run at the same time without writing to the same file.

Requires:  pip install youtube-transcript-api pandas tqdm
Uses the current (1.x) API of youtube-transcript-api.
"""
 
import argparse
import json
import os
import random
import re
import time
 
import pandas as pd
import requests
from tqdm import tqdm
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    CouldNotRetrieveTranscript,
    IpBlocked,
    RequestBlocked,
)
from youtube_transcript_api.proxies import GenericProxyConfig

MIN_DELAY, MAX_DELAY = 5.0, 10.0   # random wait between videos, in seconds
BLOCK_COOLDOWN = 3600              # pause after a block; these usually last well over 30 min
MAX_BLOCKS_IN_A_ROW = 5            # stop after this many blocks with no success between
 
 
# Errors that mean "this video genuinely has no usable transcript". These are
# cached for good. Any other error (blocks, network failures, YouTube changes
# the library can't handle yet) is saved too, but retried on the next run.
PERMANENT_ERRORS = {
    "NoTranscriptsAvailable",   # video has zero caption tracks
    "TranscriptsDisabled",
    "NoTranscriptFound",
    "VideoUnavailable",
    "VideoUnplayable",
    "AgeRestricted",
    "InvalidVideoId",
}
 
 
class Blocked(Exception):
    """YouTube refused the request because of this IP."""


class ConnectionLost(Exception):
    """Couldn't reach YouTube (or the proxy) even after retrying."""


# Network problems on our side, not anything about the video. Saving these to the
# cache would record a bogus result, so the run stops instead. ProxyError and
# SSLError are subclasses of ConnectionError.
CONNECTION_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def base_language(code):
    """'en-US' -> 'en'; None, '' or a blank CSV cell (NaN) -> None."""
    if not isinstance(code, str):
        return None
    return code.split("-")[0].lower() or None


def expected_languages(df, id_col):
    """Video ID -> the language its transcript should be in, if the CSV says."""
    expected = pd.Series(None, index=df.index, dtype=object)
    for column in ("detected_language", "default_audio_language"):  # later one wins
        if column in df.columns:
            expected = df[column].map(base_language).where(lambda v: v.notna(), expected)
    # pandas stores missing values as NaN; use None so "unknown" is handled as unknown.
    return {vid: (lang if isinstance(lang, str) else None)
            for vid, lang in zip(df[id_col].astype(str), expected)}


def make_api(proxy=None):
    if proxy:
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy,
                                                                    https_url=proxy))
    return YouTubeTranscriptApi()


def add_video_ids(df):
    """Return df without leftover index columns and with a `video_id` column."""
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])
    if "video_id" in df.columns:
        return df
    if "Video ID" in df.columns:
        return df.rename(columns={"Video ID": "video_id"})
    if "url" not in df.columns:
        raise SystemExit("The CSV needs a video_id, Video ID or url column.")
    ids = df["url"].astype(str).str.extract(r"[?&]v=([\w-]{11})", expand=False)
    if ids.isna().any():
        raise SystemExit(f"Could not read a video ID from {ids.isna().sum()} URL(s), "
                         f"e.g. {df.loc[ids.isna(), 'url'].iloc[0]}")
    df = df.copy()
    df.insert(0, "video_id", ids)
    return df


def pick_default_transcript(transcript_list, expected=None):
    """
    Choose the transcript in the video's original spoken language.
 
    0. If the video's language is known (`expected`), use a track in that language,
       preferring a manually uploaded one. Dubbed audio tracks get their own
       auto-generated captions, so the first generated track can be a dub.
    1. YouTube's auto-generated captions come from speech recognition on the
       audio, so the auto-generated track tells us the spoken language.
       If a manually uploaded track exists in that same language, prefer it
       (human-made captions are more accurate); otherwise use the generated one.
    2. If there's no auto-generated track at all, fall back to the first
       manually uploaded track, which is usually the creator's own language.
    """
    tracks = list(transcript_list)
    if not tracks:
        return None
 
    manual = [t for t in tracks if not t.is_generated]
    generated = [t for t in tracks if t.is_generated]
 
    if expected:
        for t in manual + generated:
            if base_language(t.language_code) == expected:
                return t

    if generated:
        spoken = generated[0]
        base = spoken.language_code.split("-")[0]
        for t in manual:
            if t.language_code.split("-")[0] == base:
                return t
        return spoken
 
    return manual[0]
 
 
def fetch_transcript(ytt_api, video_id, expected=None, max_retries=3):
    """
    Return (text, language_code, is_generated, error) for one video's default
    transcript. On success error is None; on failure text/language/is_generated
    are None and error names what went wrong. Raises Blocked if YouTube blocks the IP,
    and ConnectionLost if the connection keeps failing after `max_retries` tries.
    """
    last_error = None
    connection_failed = False
    for attempt in range(max_retries):
        try:
            transcript = pick_default_transcript(ytt_api.list(video_id), expected)
            if transcript is None:
                return None, None, None, "NoTranscriptsAvailable"
            fetched = transcript.fetch()
            text = " ".join(s.text.replace("\n", " ") for s in fetched)
            return text, transcript.language_code, transcript.is_generated, None
        except RequestBlocked as e:  # includes IpBlocked
            raise Blocked(type(e).__name__) from e
        except CouldNotRetrieveTranscript as e:
            # The library's own error types: TranscriptsDisabled, VideoUnavailable,
            # PoTokenRequired, YouTubeRequestFailed, etc. The class name is the reason.
            return None, None, None, type(e).__name__
        except CONNECTION_ERRORS as e:  # retry briefly; stop the run if it persists
            last_error = f"{type(e).__name__}: {str(e).strip()[:200]}"
            connection_failed = True
            tqdm.write(f"[{video_id}] connection error (try {attempt + 1}/{max_retries}): "
                       f"{last_error}")
            time.sleep(5 * (attempt + 1))
        except Exception as e:  # anything unexpected
            last_error = f"{type(e).__name__}: {str(e).strip()[:200]}"
            connection_failed = False
            tqdm.write(f"[{video_id}] unexpected error: {last_error}")
            time.sleep(5 * (attempt + 1))
    if connection_failed:
        raise ConnectionLost(last_error)
    return None, None, None, last_error


def add_transcripts(df, id_col="video_id", cache_path="transcript_cache.jsonl",
                    desc="Transcripts", proxy=None, min_delay=MIN_DELAY, max_delay=MAX_DELAY,
                    block_cooldown=BLOCK_COOLDOWN, max_blocks=MAX_BLOCKS_IN_A_ROW):
    """
    Fetch the default transcript for every video in df[id_col] and return
    (df, stopped), where df has `transcript`, `transcript_language`,
    `transcript_is_generated` and `transcript_error` columns, and stopped is True
    if the run ended early because YouTube kept blocking it or the connection failed.

    Results are cached to `cache_path` as they come in, so re-running picks up where
    it left off. Videos that failed for a temporary reason are retried, and so are
    transcripts saved by older versions of this script in the wrong language.
    Videos not reached yet have transcript_error "NotFetchedYet".
    """
    ytt_api = make_api(proxy)
    expected = expected_languages(df, id_col)

    # Load anything already fetched on a previous run. Later lines win, so a
    # successful retry replaces an earlier failure for the same video.
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line from a sudden stop; refetch it
                cache[rec["video_id"]] = rec

    def done(vid):
        """Has this video been fetched with a final result?"""
        rec = cache.get(vid)
        if rec is None:
            return False
        if rec.get("transcript") is not None:
            # Records saved with the language check ("checked") are final. Older ones
            # are final only if they're already in the expected language.
            want = expected.get(vid)
            return (rec.get("checked") or want is None
                    or base_language(rec.get("language")) == want)
        # A failure that won't change by retrying. Entries from older versions of
        # this script have no error recorded, so they get refetched.
        return rec.get("error") in PERMANENT_ERRORS

    unique_ids = df[id_col].dropna().astype(str).unique()
    todo = [vid for vid in unique_ids if not done(vid)]
    stopped = False

    with open(cache_path, "a", encoding="utf-8") as cache_file:
        pbar = tqdm(todo, desc=desc, unit="video")
        found = missing = other_language = blocks = 0
        for vid in pbar:
            while True:
                try:
                    text, lang, is_gen, error = fetch_transcript(ytt_api, vid, expected.get(vid))
                    blocks = 0
                    break
                except ConnectionLost as e:
                    # Not cached, so this video is fetched again on the next run.
                    tqdm.write(f"[{vid}] connection failed, stopping: {e}. "
                               "Progress is saved; check your internet/proxy and rerun.")
                    stopped = True
                    break
                except Blocked as e:
                    blocks += 1
                    if blocks >= max_blocks:
                        tqdm.write(f"[{vid}] {e}: blocked {blocks} times in a row, stopping. "
                                   "Progress is saved; run again later or use a proxy.")
                        stopped = True
                        break
                    tqdm.write(f"[{vid}] {e}: blocked by YouTube ({blocks}/{max_blocks}), "
                               f"pausing {block_cooldown / 60:.0f} min. Progress is saved.")
                    time.sleep(block_cooldown)
                    ytt_api = make_api(proxy)  # start a fresh session
            if stopped:
                break

            want = expected.get(vid)
            if text is None:
                missing += 1
            else:
                found += 1
                if want and base_language(lang) != want:
                    other_language += 1  # no track in the expected language
            cache[vid] = {"video_id": vid, "transcript": text, "language": lang,
                          "is_generated": is_gen, "error": error, "checked": True}
            # ensure_ascii=False keeps characters readable (é, ’, ا) instead of \uXXXX codes.
            cache_file.write(json.dumps(cache[vid], ensure_ascii=False) + "\n")
            cache_file.flush()
            pbar.set_postfix(found=found, missing=missing, other_language=other_language)
            time.sleep(random.uniform(min_delay, max_delay))  # irregular pacing looks less automated
        pbar.close()

    not_fetched = {"transcript": None, "language": None, "is_generated": None,
                   "error": "NotFetchedYet"}
    records = df[id_col].astype(str).map(lambda v: cache.get(v, not_fetched))
    df["transcript"] = records.map(lambda r: r["transcript"])
    df["transcript_language"] = records.map(lambda r: r["language"])
    df["transcript_is_generated"] = records.map(lambda r: r["is_generated"])
    df["transcript_error"] = records.map(lambda r: r["error"])
    return df, stopped


if __name__ == "__main__":
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Add YouTube transcripts to a video CSV.")
    parser.add_argument("input", help="CSV of videos, e.g. video_data1.csv")
    parser.add_argument("--output", help="default: <input>_with_transcripts.csv")
    parser.add_argument("--cache", help="default: <input>_transcript_cache.jsonl")
    parser.add_argument("--min-delay", type=float, default=MIN_DELAY,
                        help="shortest wait between videos, in seconds")
    parser.add_argument("--max-delay", type=float, default=MAX_DELAY,
                        help="longest wait between videos, in seconds")
    parser.add_argument("--block-cooldown", type=float, default=BLOCK_COOLDOWN,
                        help="seconds to pause after YouTube blocks a request")
    parser.add_argument("--max-blocks", type=int, default=MAX_BLOCKS_IN_A_ROW,
                        help="stop after this many blocks in a row")
    parser.add_argument("--proxy", help="proxy URL; default: TRANSCRIPT_PROXY from .env")
    parser.add_argument("--nrows", type=int, help="only process the first N rows, for testing")
    args = parser.parse_args()
    if args.max_delay < args.min_delay:
        parser.error("--max-delay must be at least --min-delay")

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    proxy = args.proxy or os.environ.get("TRANSCRIPT_PROXY") or None
    print("Using a proxy" if proxy else "No proxy: requests come from this computer's IP")

    stem = os.path.splitext(args.input)[0]
    output = args.output or f"{stem}_with_transcripts.csv"
    cache = args.cache or f"{stem}_transcript_cache.jsonl"
    CHUNK_SIZE = 50_000

    # Process in chunks so large files never have to fit in memory all at once.
    # The cache is shared across chunks, so nothing is fetched twice.
    reader = pd.read_csv(args.input, chunksize=CHUNK_SIZE, nrows=args.nrows, low_memory=False)
    stopped = False
    try:
        for i, chunk in enumerate(reader, 1):
            chunk, stopped = add_transcripts(
                add_video_ids(chunk), cache_path=cache, desc=f"Chunk {i}", proxy=proxy,
                min_delay=args.min_delay, max_delay=args.max_delay,
                block_cooldown=args.block_cooldown, max_blocks=args.max_blocks)
            chunk.to_csv(output, mode="w" if i == 1 else "a", header=(i == 1), index=False)
            print(chunk["transcript_error"].fillna("ok").value_counts().to_string())
            if stopped:
                break
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved in the cache; run the same command to resume.")
        raise SystemExit(130)

    if stopped:
        print(f"Stopped early (repeated blocks or a lost connection). {output} has what's done so far; "
              "run the same command later to continue.")
    else:
        print(f"Saved {output}")
