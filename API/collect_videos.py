"""
Build a balanced dataset of about 100,000 English YouTube videos with the
YouTube Data API v3.

Design:
    14 genres x 4 subscriber tiers = 56 groups
    72 channels per group x 25 videos per channel = 100,800 sampled videos

Music is left out on purpose, and Shorts are excluded by the 3-minute minimum.

Steps:
    1. Discover channels with search.list (100 units per call).
    2. Check each channel with channels.list and assign it to a group.
    3. List its recent uploads with playlistItems.list.
    4. Get full video metadata with videos.list and apply the filters.
    5. Accept the channel if at least 25 videos pass, and sample 25 of them.

Steps 1-5 repeat until every group is full or --max-searches is reached.

Keys:
    Reads YOUTUBE_API1, YOUTUBE_API2, ... from .env and moves to the next key
    when one runs out of quota. Quota belongs to the Google Cloud project, so
    keys from the same project share one daily allowance.

Setup:
    uv pip install --python .venv/bin/python requests pandas tqdm \
        python-dotenv openpyxl langdetect

Run:
    .venv/bin/python collect_videos.py
    .venv/bin/python collect_videos.py --max-searches 2 --channels-per-group 1 \
        --out-dir dataset_test

Resuming:
    Every result is saved as it arrives. If the keys run out or the run stops,
    run the same command again (quota resets at midnight Pacific time).

Output (in dataset/):
    videos_sample.csv / .xlsx   the training sample (in_sample == True)
    videos_all.csv              every eligible video, for channel baselines
    channels.jsonl              every channel checked, with its status
    searches.jsonl              every search made
    channel_results.jsonl       accept/reject result for each channel tried
    videos.jsonl                raw rows behind the CSV files
"""

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
GENRES = {  # official name -> videoCategoryId: every assignable category except Music (10)
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Pets & Animals": "15",
    "Sports": "17",
    "Travel & Events": "19",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
    "Nonprofits & Activism": "29",
}
TIERS = [  # label, minimum subscribers (inclusive), maximum (exclusive)
    ("1k-10k", 1_000, 10_000),
    ("10k-100k", 10_000, 100_000),
    ("100k-1M", 100_000, 1_000_000),
    ("1M+", 1_000_000, float("inf")),
]
SMALL_TIERS = {"1k-10k", "10k-100k"}

SEARCH_TERMS = {
    "Film & Animation": ["", "short film", "animation", "movie review", "film analysis",
                         "behind the scenes", "video essay", "filmmaking", "cartoon",
                         "movie recap"],
    "Autos & Vehicles": ["", "car review", "test drive", "car repair", "motorcycle",
                         "truck", "restoration", "electric car", "racing", "detailing"],
    "Pets & Animals": ["", "dog training", "cat", "wildlife", "pet care", "aquarium",
                       "farm animals", "birds", "rescue", "veterinarian"],
    "Sports": ["", "highlights", "training", "football", "basketball", "soccer",
               "boxing", "running", "game recap", "interview"],
    "Travel & Events": ["", "travel vlog", "travel guide", "road trip", "hotel review",
                        "backpacking", "things to do", "van life", "city tour", "cruise"],
    "Gaming": ["", "gameplay", "walkthrough", "let's play", "review", "minecraft",
               "indie game", "speedrun", "strategy", "stream highlights"],
    "People & Blogs": ["", "vlog", "day in my life", "storytime", "podcast", "q&a",
                       "my story", "family vlog", "advice", "daily routine"],
    "Comedy": ["", "stand up", "sketch", "funny", "parody", "skit", "improv",
               "roast", "prank", "comedy podcast"],
    "Entertainment": ["", "reaction", "celebrity interview", "tv show", "challenge",
                      "commentary", "talk show", "reality tv", "movie trailer reaction",
                      "top 10"],
    "News & Politics": ["", "news", "analysis", "politics", "local news", "world news",
                        "interview", "opinion", "explained", "breaking"],
    "Howto & Style": ["", "how to", "tutorial", "diy", "recipe", "makeup", "home repair",
                      "woodworking", "gardening", "guide"],
    "Education": ["", "lecture", "explained", "history", "science", "math",
                  "language learning", "study tips", "documentary", "course"],
    "Science & Technology": ["", "review", "technology", "programming", "ai", "engineering",
                             "space", "unboxing", "electronics", "how it works"],
    "Nonprofits & Activism": ["", "charity", "volunteer", "climate action", "fundraiser",
                              "human rights", "community", "awareness", "nonprofit",
                              "activism"],
}
REGIONS = ["US", "GB", "CA", "AU", "IE", "NZ"]

CANDIDATE_MULT = 1.6      # keep 1.6x the channels still needed, since some get rejected
MAX_UPLOAD_PAGES = 2      # uploads pages per channel (50 videos each)
LOOKBACK_DAYS = 3 * 365   # search for uploads from the last 3 years
MIN_CHANNEL_VIDEOS = 50   # channels with fewer uploads can't support a baseline
MIN_AGE_DAYS = 30         # newer videos' ratios are still changing
MIN_DURATION_SECONDS = 181  # the API has no Shorts flag, but Shorts can't exceed 3 minutes
MIN_VIEWS = 100           # ratios on very few views are mostly noise
MAX_RETRIES = 6

BASE_URL = "https://www.googleapis.com/youtube/v3"
QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
BAD_KEY_REASONS = {"keyInvalid", "keyExpired", "accessNotConfigured", "ipRefererBlocked"}
RETRY_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "backendError",
                 "internalError"}

_write_lock = threading.Lock()


class QuotaExhausted(Exception):
    """Every API key is out of quota or unusable."""


# ----------------------------------------------------------------------------
# API access
# ----------------------------------------------------------------------------
def load_keys():
    keys = []
    for name, value in os.environ.items():
        if re.fullmatch(r"YOUTUBE_API\d+", name) and value.strip():
            keys.append((int(name.removeprefix("YOUTUBE_API")), value.strip()))
    unique = list(dict.fromkeys(value for _, value in sorted(keys)))
    if not unique:
        raise SystemExit("No keys found. Add YOUTUBE_API1, YOUTUBE_API2, ... to .env.")
    return unique


class KeyPool:
    """Calls the API, moving to the next key when one is used up."""

    def __init__(self, keys):
        self.keys = keys
        self.index = 0
        self.units = Counter()
        self.lock = threading.Lock()
        self.session = requests.Session()

    def _current(self):
        with self.lock:
            if self.index >= len(self.keys):
                raise QuotaExhausted
            return self.index

    def _advance(self, index, reason):
        with self.lock:
            if self.index == index:  # another thread may have moved on already
                self.index += 1
                tqdm.write(f"Key {index + 1}/{len(self.keys)} stopped ({reason}) "
                           f"after ~{self.units[index]:,} units this run")

    def call(self, endpoint, params, cost=1):
        """Return the JSON response, or None if the request can't succeed."""
        attempts = 0
        while attempts < MAX_RETRIES:
            index = self._current()
            try:
                # The key goes in a header so it can't end up in a logged URL.
                resp = self.session.get(f"{BASE_URL}/{endpoint}", params=params,
                                        headers={"X-Goog-Api-Key": self.keys[index]},
                                        timeout=30)
            except requests.RequestException as e:
                attempts += 1
                tqdm.write(f"{endpoint}: {type(e).__name__}, retrying")
                time.sleep(2 ** attempts)
                continue

            if resp.status_code == 200:
                with self.lock:
                    self.units[index] += cost
                return resp.json()

            try:
                error = resp.json().get("error", {})
                reason = (error.get("errors") or [{}])[0].get("reason", "")
                message = error.get("message", "")
            except ValueError:
                reason, message = "", resp.text[:200]

            # Google reports an invalid key as a generic badRequest, and a used-up
            # daily search quota as a 429 rateLimitExceeded ("... per day").
            daily_limit = "per day" in message.lower()
            if (reason in QUOTA_REASONS or reason in BAD_KEY_REASONS or daily_limit
                    or "API key" in message):
                self._advance(index, "daily limit" if daily_limit else reason or "bad key")
            elif reason in RETRY_REASONS or resp.status_code == 429 or resp.status_code >= 500:
                attempts += 1
                time.sleep(2 ** attempts)
            elif resp.status_code == 404:
                return None  # e.g. a deleted uploads playlist
            else:
                tqdm.write(f"{endpoint}: {resp.status_code} {reason}: {message[:200]}")
                return None
        tqdm.write(f"{endpoint}: giving up after {MAX_RETRIES} attempts")
        return None


# ----------------------------------------------------------------------------
# Saved state
# ----------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # half-written last line from a sudden stop
    return rows


def append_jsonl(path, rows):
    with _write_lock, open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class State:
    """Everything done so far, rebuilt from the output folder on each run."""

    def __init__(self, out_dir, channels_per_group, max_dry_searches=5):
        self.searches_path = os.path.join(out_dir, "searches.jsonl")
        self.channels_path = os.path.join(out_dir, "channels.jsonl")
        self.results_path = os.path.join(out_dir, "channel_results.jsonl")
        self.videos_path = os.path.join(out_dir, "videos.jsonl")
        self.target = channels_per_group
        self.max_dry = max_dry_searches

        searches = load_jsonl(self.searches_path)
        channels = load_jsonl(self.channels_path)
        self.searches = {r["query_key"] for r in searches}
        self.channels = {r["channel_id"]: r for r in channels}
        self.results = {r["channel_id"]: r for r in load_jsonl(self.results_path)}
        self.dry = self._dry_streaks(searches, channels)

    @staticmethod
    def _dry_streaks(searches, channels):
        """Count each group's latest run of searches that found no new candidates for it."""
        dry = Counter()
        position = 0
        for search in searches:
            # Each search's channels were saved just before it, so they line up in order.
            found = channels[position:position + search["new_channels"]]
            position += search["new_channels"]
            group = State.group(search["genre"], search["tier"])
            if "target_candidates" in search:
                count = search["target_candidates"]
            elif search["results"] > 0:
                count = sum(c["status"] == "candidate" and c["group"] == group for c in found)
            else:
                continue  # older runs saved failed API calls as empty searches; ignore them
            dry[group] = 0 if count else dry[group] + 1
        return dry

    def exhausted(self, group):
        """True once a group's searches keep finding nothing new."""
        return self.dry[group] >= self.max_dry

    @staticmethod
    def group(genre, tier):
        return f"{genre} | {tier}"

    def accepted(self, group):
        return sum(1 for cid, r in self.results.items()
                   if r["status"] == "accepted" and self.channels[cid]["group"] == group)

    def untried(self, group):
        return [c for cid, c in self.channels.items()
                if c.get("status") == "candidate" and c["group"] == group
                and cid not in self.results]

    def all_groups(self):
        return [self.group(g, t[0]) for g in GENRES for t in TIERS]

    def needs_channels(self, group):
        """True if this group needs more candidates from search."""
        missing = self.target - self.accepted(group)
        return (missing > 0 and not self.exhausted(group)
                and len(self.untried(group)) < CANDIDATE_MULT * missing)

    def complete(self):
        return all(self.accepted(g) >= self.target for g in self.all_groups())


# ----------------------------------------------------------------------------
# Channel discovery
# ----------------------------------------------------------------------------
def make_query(genre, tier, used, rng):
    """Build a search not used before. Varying these gets past the ~500 result cap."""
    now = datetime.now(timezone.utc)
    for _ in range(100):
        window = 7 if tier in SMALL_TIERS else 30
        end = now - timedelta(days=rng.randint(MIN_AGE_DAYS, LOOKBACK_DAYS))
        start = end - timedelta(days=window)
        query = {
            "q": rng.choice(SEARCH_TERMS[genre]),
            "regionCode": rng.choice(REGIONS),
            # Newest-first reaches small channels; most-viewed reaches big ones.
            "order": "date" if tier in SMALL_TIERS else "viewCount",
            "publishedAfter": start.strftime("%Y-%m-%dT00:00:00Z"),
            "publishedBefore": end.strftime("%Y-%m-%dT00:00:00Z"),
        }
        key = "|".join([genre, *query.values()])
        if key not in used:
            return key, query
    raise RuntimeError("Could not build a new search query")


def tier_for(subscribers):
    for label, low, high in TIERS:
        if low <= subscribers < high:
            return label
    return None


def channel_record(item, genre):
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    subscribers = int(stats.get("subscriberCount", 0))
    videos = int(stats.get("videoCount", 0))
    uploads = item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    tier = tier_for(subscribers)

    if stats.get("hiddenSubscriberCount"):
        status = "rejected_hidden_subscribers"
    elif snippet.get("title", "").endswith(" - Topic"):
        status = "rejected_auto_generated"
    elif item.get("status", {}).get("madeForKids"):
        status = "rejected_made_for_kids"
    elif tier is None:
        status = "rejected_too_small"
    elif videos < MIN_CHANNEL_VIDEOS:
        status = "rejected_too_few_uploads"
    elif not uploads:
        status = "rejected_no_uploads_playlist"
    else:
        status = "candidate"

    return {
        "channel_id": item["id"],
        "status": status,
        "genre": genre,
        "size_tier": tier,
        "group": State.group(genre, tier) if tier else None,
        "uploads_playlist": uploads,
        "channel_title": snippet.get("title"),
        "channel_description": snippet.get("description"),
        "channel_custom_url": snippet.get("customUrl"),
        "channel_country": snippet.get("country"),
        "channel_default_language": snippet.get("defaultLanguage"),
        "channel_published_at": snippet.get("publishedAt"),
        "channel_subscribers": subscribers,
        "channel_views": int(stats.get("viewCount", 0)),
        "channel_video_count": videos,
        "channel_topics": "|".join(item.get("topicDetails", {}).get("topicCategories", [])),
    }


def discover(pool, state, max_searches, rng):
    """Search round-robin across groups that still need channels."""
    pbar = tqdm(total=max_searches, initial=min(len(state.searches), max_searches),
                desc="Searches", unit="call")
    failures = 0
    while len(state.searches) < max_searches:
        cells = [(g, t[0]) for g in GENRES for t in TIERS
                 if state.needs_channels(state.group(g, t[0]))]
        if not cells:
            break
        for genre, tier in cells:
            if len(state.searches) >= max_searches:
                break
            if not state.needs_channels(state.group(genre, tier)):
                continue

            key, query = make_query(genre, tier, state.searches, rng)
            params = {"part": "snippet", "type": "video", "maxResults": 50,
                      "videoCategoryId": GENRES[genre], "relevanceLanguage": "en",
                      "fields": "items/snippet/channelId",
                      **{k: v for k, v in query.items() if v}}
            data = pool.call("search", params, cost=100)
            if data is None:
                # Failed calls aren't saved, so they don't use up --max-searches.
                failures += 1
                if failures >= 10:
                    tqdm.write("10 searches in a row failed; stopping the search stage.")
                    pbar.close()
                    return
                continue
            failures = 0
            items = data.get("items", [])

            new_ids = list(dict.fromkeys(i["snippet"]["channelId"] for i in items))
            new_ids = [cid for cid in new_ids if cid not in state.channels]
            records = []
            if new_ids:
                info = pool.call("channels", {
                    "part": "snippet,statistics,contentDetails,status,topicDetails",
                    "id": ",".join(new_ids), "maxResults": 50})
                found = {i["id"]: i for i in (info or {}).get("items", [])}
                records = [channel_record(found[cid], genre) for cid in new_ids if cid in found]

            for record in records:
                state.channels[record["channel_id"]] = record
            append_jsonl(state.channels_path, records)

            group = state.group(genre, tier)
            target_candidates = sum(r["status"] == "candidate" and r["group"] == group
                                    for r in records)
            state.dry[group] = 0 if target_candidates else state.dry[group] + 1
            if state.exhausted(group):
                tqdm.write(f"No longer searching {group}: {state.max_dry} searches in a row "
                           "found no new candidates")

            state.searches.add(key)
            append_jsonl(state.searches_path, [{
                "query_key": key, "genre": genre, "tier": tier, **query,
                "results": len(items), "new_channels": len(records),
                "new_candidates": sum(r["status"] == "candidate" for r in records),
                "target_candidates": target_candidates}])
            pbar.update(1)
    pbar.close()


# ----------------------------------------------------------------------------
# Video collection
# ----------------------------------------------------------------------------
def parse_duration(value):
    """Convert an ISO 8601 duration such as PT1H2M3S to seconds."""
    match = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?", value or "")
    if not match:
        return None
    days, hours, minutes, seconds = (int(x or 0) for x in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def detect_language(title, description):
    text = re.sub(r"https?://\S+", " ", f"{title or ''}. {description or ''}")[:500]
    try:
        return detect(text)
    except LangDetectException:
        return None  # nothing detectable, e.g. only symbols


def video_row(item, channel, collected_at):
    snippet = item.get("snippet", {})
    stats = item.get("statistics", {})
    details = item.get("contentDetails", {})
    status = item.get("status", {})
    views, likes, comments = (int(stats[k]) if k in stats else None
                              for k in ("viewCount", "likeCount", "commentCount"))
    published = snippet.get("publishedAt")
    age_days = None
    if published:
        age = collected_at - datetime.fromisoformat(published.replace("Z", "+00:00"))
        age_days = round(age.total_seconds() / 86400, 2)

    return {
        "video_id": item["id"],
        "url": f"https://www.youtube.com/watch?v={item['id']}",
        "group": channel["group"],
        "genre": channel["genre"],
        "size_tier": channel["size_tier"],
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "tags": "|".join(snippet.get("tags", [])),
        "published_at": published,
        "age_days": age_days,
        "category_id": snippet.get("categoryId"),
        "default_language": snippet.get("defaultLanguage"),
        "default_audio_language": snippet.get("defaultAudioLanguage"),
        "detected_language": None,
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
        "duration_seconds": parse_duration(details.get("duration")),
        "definition": details.get("definition"),
        "has_captions": details.get("caption") == "true",
        "licensed_content": details.get("licensedContent"),
        "live_broadcast_content": snippet.get("liveBroadcastContent"),
        "privacy_status": status.get("privacyStatus"),
        "made_for_kids": status.get("madeForKids"),
        "topics": "|".join(item.get("topicDetails", {}).get("topicCategories", [])),
        "views": views,
        "likes": likes,
        "comments": comments,
        "like_view_ratio": likes / views if views and likes is not None else None,
        "comment_view_ratio": comments / views if views and comments is not None else None,
        **{k: channel[k] for k in (
            "channel_id", "channel_title", "channel_custom_url", "channel_country",
            "channel_default_language", "channel_published_at", "channel_subscribers",
            "channel_views", "channel_video_count", "channel_topics")},
        "collected_at": collected_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def is_music(topics):
    """True if YouTube tags this as music, e.g. .../wiki/Music or .../wiki/Hip_hop_music."""
    return any(t.rsplit("/", 1)[-1].lower().endswith("music") for t in topics.split("|"))


def rejection_reason(row, genre_id):
    """Return why a video fails the filters, or None if it passes. Language is last (slowest)."""
    if row["category_id"] != genre_id:
        return "wrong_category"
    if is_music(row["topics"]):
        return "music"
    if row["privacy_status"] != "public" or row["live_broadcast_content"] != "none":
        return "not_public_or_live"
    if row["made_for_kids"]:
        return "made_for_kids"
    if row["views"] is None or row["likes"] is None or row["comments"] is None:
        return "hidden_stats"  # likes hidden or comments disabled
    if row["views"] < MIN_VIEWS:
        return "too_few_views"
    if (row["age_days"] or 0) < MIN_AGE_DAYS:
        return "too_new"
    if (row["duration_seconds"] or 0) < MIN_DURATION_SECONDS:
        return "too_short"

    row["detected_language"] = detect_language(row["title"], row["description"])
    language = row["default_audio_language"] or row["detected_language"] or ""
    return None if language.split("-")[0].lower() == "en" else "not_english"


def process_channel(pool, channel, videos_per_channel):
    """Fetch and filter one channel's uploads. Returns (result, video rows)."""
    genre_id = GENRES[channel["genre"]]
    collected_at = datetime.now(timezone.utc)
    eligible, seen, page_token = [], 0, None
    dropped = Counter()

    for _ in range(MAX_UPLOAD_PAGES):
        params = {"part": "contentDetails", "playlistId": channel["uploads_playlist"],
                  "maxResults": 50, "fields": "nextPageToken,items/contentDetails/videoId"}
        if page_token:
            params["pageToken"] = page_token
        page = pool.call("playlistItems", params)
        if not page:
            break
        ids = [i["contentDetails"]["videoId"] for i in page.get("items", [])]
        if ids:
            data = pool.call("videos", {
                "part": "snippet,contentDetails,statistics,status,topicDetails",
                "id": ",".join(ids), "maxResults": 50})
            for item in (data or {}).get("items", []):
                seen += 1
                row = video_row(item, channel, collected_at)
                reason = rejection_reason(row, genre_id)
                if reason:
                    dropped[reason] += 1
                else:
                    eligible.append(row)
        page_token = page.get("nextPageToken")
        if len(eligible) >= videos_per_channel or not page_token:
            break

    result = {"channel_id": channel["channel_id"], "videos_seen": seen,
              "videos_eligible": len(eligible), "videos_dropped": dict(dropped)}
    if len(eligible) < videos_per_channel:
        return {**result, "status": "rejected_too_few_eligible"}, []

    chosen = set(random.Random(channel["channel_id"]).sample(
        [r["video_id"] for r in eligible], videos_per_channel))
    for row in eligible:
        row["in_sample"] = row["video_id"] in chosen
    return {**result, "status": "accepted"}, eligible


def collect_videos(pool, state, videos_per_channel, workers, rng):
    """Try candidates in parallel until each group is full or runs out."""
    pbar = tqdm(total=state.target * len(state.all_groups()),
                initial=sum(min(state.accepted(g), state.target) for g in state.all_groups()),
                desc="Channels accepted", unit="channel")
    while True:
        batch = []
        for group in state.all_groups():
            missing = state.target - state.accepted(group)
            if missing > 0:
                candidates = state.untried(group)
                rng.shuffle(candidates)
                batch.extend(candidates[:missing])
        if not batch:
            break

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_channel, pool, c, videos_per_channel)
                       for c in batch]
            stop = None
            for future in as_completed(futures):
                try:
                    result, rows = future.result()
                except QuotaExhausted as e:
                    stop = e  # finish saving whatever the other threads completed
                    continue
                append_jsonl(state.videos_path, rows)
                append_jsonl(state.results_path, [result])
                state.results[result["channel_id"]] = result
                if result["status"] == "accepted":
                    pbar.update(1)
        if stop:
            pbar.close()
            raise stop
    pbar.close()


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def to_excel_safe(df):
    """Remove characters Excel rejects, and stop text being read as a formula."""
    illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def clean(value):
        if not isinstance(value, str):
            return value
        value = illegal.sub(" ", value)[:32767]
        return "'" + value if value.startswith(("=", "+", "-", "@")) else value

    df = df.copy()
    for column in df.columns:
        if df[column].dtype == object or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].map(clean)
    return df


def build_dataframe(out_dir):
    """Return the full DataFrame of eligible videos, one row per video."""
    rows = load_jsonl(os.path.join(out_dir, "videos.jsonl"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("video_id", keep="last")
    return df.reset_index(drop=True)


def save_outputs(out_dir, state, excel=True):
    df = build_dataframe(out_dir)
    if df.empty:
        print("No videos collected yet.")
        return df

    sample = df[df["in_sample"]]
    df.to_csv(os.path.join(out_dir, "videos_all.csv"), index=False)
    sample.to_csv(os.path.join(out_dir, "videos_sample.csv"), index=False)
    if excel:
        to_excel_safe(sample).to_excel(os.path.join(out_dir, "videos_sample.xlsx"), index=False)

    print(f"\nSaved {len(sample):,} sampled videos and {len(df):,} total eligible videos "
          f"from {df['channel_id'].nunique():,} channels to {out_dir}/")
    print(f"\nChannels accepted per group (target {state.target}):")
    counts = {g: state.accepted(g) for g in state.all_groups()}
    for genre in GENRES:
        cells = "  ".join(f"{t[0]}: {counts[State.group(genre, t[0])]:>4}" for t in TIERS)
        print(f"  {genre:<22} {cells}")
    return df


def print_status(state, max_searches):
    short = [g for g in state.all_groups() if state.accepted(g) < state.target]
    print(f"\nSearches used: {len(state.searches):,} of {max_searches:,} allowed")
    exhausted = [g for g in short if state.exhausted(g)]
    if exhausted:
        print(f"No longer searched ({state.max_dry} searches in a row found nothing new; "
              "raise --max-dry-searches to try again):")
        for group in exhausted:
            print(f"  {group}: {state.accepted(group)} accepted, "
                  f"{len(state.untried(group))} candidates untried")
    if short:
        print(f"{len(short)} group(s) are still short of {state.target} channels. "
              "Run the same command again, raising --max-searches if that limit was reached.")
    else:
        print("All groups are full.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--channels-per-group", type=int, default=72)
    parser.add_argument("--videos-per-channel", type=int, default=25)
    parser.add_argument("--max-searches", type=int, default=700,
                        help="total search.list calls across all runs (100 units each)")
    parser.add_argument("--max-dry-searches", type=int, default=5,
                        help="stop searching a group after this many searches in a row "
                             "find no new candidates for it")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", default="dataset")
    parser.add_argument("--no-excel", action="store_true", help="skip the .xlsx file")
    args = parser.parse_args()

    load_dotenv()
    os.makedirs(args.out_dir, exist_ok=True)
    DetectorFactory.seed = 0  # makes language detection repeatable
    detect("this loads the language profiles before the worker threads start")

    keys = load_keys()
    print(f"Loaded {len(keys)} API key(s)")
    pool = KeyPool(keys)
    state = State(args.out_dir, args.channels_per_group, args.max_dry_searches)
    rng = random.Random()
    print(f"Resuming with {len(state.searches):,} searches, {len(state.channels):,} channels "
          f"and {len(state.results):,} channels tried")
    for group in state.all_groups():
        if state.exhausted(group):
            print(f"  Not searching {group} ({state.dry[group]} dry searches in a row)")

    try:
        while not state.complete():
            before = (len(state.searches), len(state.results))
            discover(pool, state, args.max_searches, rng)
            collect_videos(pool, state, args.videos_per_channel, args.workers, rng)
            if (len(state.searches), len(state.results)) == before:
                break  # nothing left to try within the search budget
    except QuotaExhausted:
        print("\nAll API keys are out of quota. Run again after midnight Pacific time.")
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command to resume.")

    print(f"Approximate units used this run: {sum(pool.units.values()):,}")
    save_outputs(args.out_dir, state, excel=not args.no_excel)
    print_status(state, args.max_searches)


if __name__ == "__main__":
    main()
