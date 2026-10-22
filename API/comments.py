import os
import requests
import pandas as pd
from dotenv import load_dotenv
 
load_dotenv()
 
def fetch_comments(part, video_col="Video ID", out_col="Comment Texts",
                   num_keys=10, start_key=1, save_path=None, save_every=100):
    base = "https://www.googleapis.com/youtube/v3"
    if out_col not in part.columns:
        part[out_col] = None
    part[out_col] = part[out_col].astype(object)
 
    #start = int(part[out_col].isna().to_numpy().argmax())
    start = 0
    todo = part.iloc[start:]
    print(f"Starting at row {todo.index[0]} ({len(todo)} videos to go)")
 
    state = {"x": start_key, "key": os.environ.get(f"YOUTUBE_API{start_key}"), "done": False}
    if not state["key"]:
        print(f"YOUTUBE_API{start_key} not found. Check the key names in your .env file.")
        return part
    print(f"Using YOUTUBE_API{start_key}")
 
    def call(endpoint, params):
        if state["done"]:
            return {}
        resp = requests.get(f"{base}/{endpoint}", params={**params, "key": state["key"]})
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            error = resp.json().get("error", {})
            reason = error.get("errors", [{}])[0].get("reason")
            if reason not in ("quotaExceeded", "dailyLimitExceeded"):
                print(f"API error {resp.status_code} ({reason}) for {params.get('videoId') or params.get('parentId')}: {error.get('message')}")
                return {}
            state["x"] += 1
            if state["x"] > num_keys or not os.environ.get(f"YOUTUBE_API{state['x']}"):
                state["done"] = True
                return {}
            state["key"] = os.environ[f"YOUTUBE_API{state['x']}"]
            print(f"Switched to YOUTUBE_API{state['x']}")
            return call(endpoint, params)
        return resp.json()
 
    for i, (row, video_id) in enumerate(zip(todo.index, todo[video_col])):
        comments = []
        params = {"part": "snippet,replies", "videoId": video_id,
                  "maxResults": 100, "textFormat": "plainText"}
        while True:
            data = call("commentThreads", params)
            for thread in data.get("items", []):
                top = thread["snippet"]
                comments.append(top["topLevelComment"]["snippet"]["textDisplay"])
                inline = thread.get("replies", {}).get("comments", [])
                if top.get("totalReplyCount", 0) > len(inline):
                    rparams = {"part": "snippet", "parentId": thread["id"],
                               "maxResults": 100, "textFormat": "plainText"}
                    while True:
                        rdata = call("comments", rparams)
                        comments.extend(r["snippet"]["textDisplay"] for r in rdata.get("items", []))
                        if "nextPageToken" not in rdata:
                            break
                        rparams["pageToken"] = rdata["nextPageToken"]
                else:
                    comments.extend(r["snippet"]["textDisplay"] for r in inline)
            if "nextPageToken" not in data:
                break
            params["pageToken"] = data["nextPageToken"]
 
        if state["done"]:
            print(f"All {num_keys} keys used up. Stopped at row {row}; {len(todo) - i} videos not fetched.")
            break
        part.at[row, out_col] = comments
        print(f"{start + i + 1}/{len(part)} videos data collected")
        if save_path and (i + 1) % save_every == 0:
            part.to_csv(save_path)
            print(f"Saved {i + 1} videos to {save_path}")
 
    if save_path:
        part.to_csv(save_path)
        print(f"Saved final results to {save_path}")
    return part
 
path = 'video_data1.csv'
part = pd.read_csv(path, index_col=0)
fetch_comments(part, start_key=1, save_path=path)
 

