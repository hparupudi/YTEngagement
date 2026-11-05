"""
Fetch the original-language ("default") YouTube transcript for every video in
video_data.csv and store it in new columns:
 
    transcript               full transcript text (None if it couldn't be fetched)
    transcript_language      language code of that transcript, which doubles as
                             the video's spoken language
    transcript_is_generated  True if it's YouTube's auto-generated captions,
                             False if it was uploaded manually
    transcript_error         why there's no transcript, e.g. "TranscriptsDisabled",
                             "VideoUnavailable", "RequestBlocked" (None on success)
 
No language list is needed: each video's available tracks are listed and the
original-language one is picked automatically.
 
Requires:  pip install youtube-transcript-api pandas tqdm
Uses the current (1.x) API of youtube-transcript-api.
"""
 
import json
import os
import time
 
import pandas as pd
from tqdm import tqdm
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    CouldNotRetrieveTranscript,
    IpBlocked,
    RequestBlocked,
)
# Optional, if YouTube starts blocking your IP:
# from youtube_transcript_api.proxies import WebshareProxyConfig, GenericProxyConfig
 
 
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
 
 
def pick_default_transcript(transcript_list):
    """
    Choose the transcript in the video's original spoken language.
 
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
 
    if generated:
        spoken = generated[0]
        base = spoken.language_code.split("-")[0]
        for t in manual:
            if t.language_code.split("-")[0] == base:
                return t
        return spoken
 
    return manual[0]
 
 
def fetch_transcript(ytt_api, video_id, max_retries=3):
    """
    Return (text, language_code, is_generated, error) for one video's default
    transcript. On success error is None; on failure text/language/is_generated
    are None and error names what went wrong.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            transcript = pick_default_transcript(ytt_api.list(video_id))
            if transcript is None:
                return None, None, None, "NoTranscriptsAvailable"
            fetched = transcript.fetch()
            text = " ".join(s.text.replace("\n", " ") for s in fetched)
            return text, transcript.language_code, transcript.is_generated, None
        except (IpBlocked, RequestBlocked) as e:
            # YouTube is rate-limiting/blocking this IP: back off and retry
            last_error = type(e).__name__
            wait = 60 * (attempt + 1)
            tqdm.write(f"[{video_id}] {last_error}, sleeping {wait}s...")
            time.sleep(wait)
        except CouldNotRetrieveTranscript as e:
            # The library's own error types: TranscriptsDisabled, VideoUnavailable,
            # PoTokenRequired, YouTubeRequestFailed, etc. The class name is the reason.
            return None, None, None, type(e).__name__
        except Exception as e:  # network hiccups and anything unexpected
            last_error = f"{type(e).__name__}: {str(e).strip()[:200]}"
            tqdm.write(f"[{video_id}] unexpected error: {last_error}")
            time.sleep(5)
 
    if last_error in ("IpBlocked", "RequestBlocked"):
        raise RuntimeError(f"Still blocked after {max_retries} retries on {video_id}; "
                           "stop and try later or use a proxy.")
    return None, None, None, last_error
 
 
def add_transcripts(df, id_col="Video ID", delay=1,
                    cache_path="transcript_cache.jsonl", ytt_api=None, desc="Transcripts"):
    """
    Fetch the default transcript for every video in df[id_col] and return df
    with `transcript`, `transcript_language`, `transcript_is_generated` and
    `transcript_error` columns.
 
    Results are cached to `cache_path` as they come in, so if the run is
    interrupted (or YouTube blocks you), re-running picks up where it left off.
    Videos that failed for a temporary reason are retried on the next run.
    """
    ytt_api = ytt_api or YouTubeTranscriptApi()
    empty = (None, None, None, None)
 
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
                cache[rec["video_id"]] = (rec["transcript"], rec.get("language"),
                                          rec.get("is_generated"), rec.get("error"))
 
    def done(vid):
        """Has this video been fetched with a final result?"""
        if vid not in cache:
            return False
        text, _, _, error = cache[vid]
        # Success, or a failure that won't change by retrying. Entries from older
        # versions of this script have no error recorded, so they get refetched.
        return text is not None or error in PERMANENT_ERRORS
 
    unique_ids = df[id_col].dropna().astype(str).unique()
    todo = [vid for vid in unique_ids if not done(vid)]
 
    with open(cache_path, "a", encoding="utf-8") as cache_file:
        pbar = tqdm(todo, desc=desc, unit="video")
        found = missing = 0
        for vid in pbar:
            text, lang, is_gen, error = fetch_transcript(ytt_api, vid)
            cache[vid] = (text, lang, is_gen, error)
            if text is None:
                missing += 1
            else:
                found += 1
            cache_file.write(json.dumps({"video_id": vid, "transcript": text,
                                         "language": lang, "is_generated": is_gen,
                                         "error": error}) + "\n")
            cache_file.flush()
            pbar.set_postfix(found=found, missing=missing)
            time.sleep(delay)  # be polite; lowers the chance of an IP block
 
    ids = df[id_col].astype(str)
    df["transcript"] = ids.map(lambda v: cache.get(v, empty)[0])
    df["transcript_language"] = ids.map(lambda v: cache.get(v, empty)[1])
    df["transcript_is_generated"] = ids.map(lambda v: cache.get(v, empty)[2])
    df["transcript_error"] = ids.map(lambda v: cache.get(v, empty)[3])
    return df
 
 
if __name__ == "__main__":
    INPUT = "video_data1.csv"
    OUTPUT = "video_data1_with_transcripts.csv"
    CHUNK_SIZE = 50_000
 
    # video_data.csv is ~2.8 GB, so process it in chunks instead of all at once.
    # The cache file is shared across chunks, so nothing is fetched twice.
    reader = pd.read_csv(INPUT, index_col=0, chunksize=CHUNK_SIZE)
    for i, chunk in enumerate(reader, 1):
        chunk = add_transcripts(chunk, desc=f"Chunk {i}")
        chunk.to_csv(OUTPUT, mode="w" if i == 1 else "a", header=(i == 1))
 
    # For a small sample instead:
    # df = pd.read_csv(INPUT, index_col=0, nrows=50)
    # df = add_transcripts(df)
    # print(df["transcript_error"].value_counts(dropna=False))
