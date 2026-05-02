#!/usr/bin/env python3
"""Download all episode replays and agent logs for a Kaggle submission.

The official ``kaggle competitions replay`` path can crash with::

    KeyError: 'content-length'

when the API serves chunked JSON (no Content-Length). This script uses the
same RPCs as the CLI but streams the HTTP body to disk.

Bursts of many episodes can trigger HTTP 429; this script retries with backoff
and optional pacing. Re-run the same command to resume (skips non-empty files).

Usage::

    uv run python scripts/download_submission_replays_logs.py <submission_ref> [logs_subdir]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetEpisodeAgentLogsRequest,
    ApiGetEpisodeReplayRequest,
)


def _stream_response_to_file(response: requests.Response, outfile: str) -> None:
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(outfile, "wb") as out:
            for chunk in response.iter_content(chunk_size=512 * 1024):
                if chunk:
                    out.write(chunk)
    finally:
        response.close()


def _backoff_seconds(attempt: int, response: requests.Response | None) -> float:
    """Seconds to wait before retry. Honors Retry-After when numeric."""
    if response is not None:
        ra = response.headers.get("Retry-After")
        if ra is not None:
            try:
                return min(float(ra), 120.0)
            except ValueError:
                pass
    return min(2.0**attempt + random.random(), 90.0)


def _agent_logs_request(episode_id: int, agent_index: int) -> ApiGetEpisodeAgentLogsRequest:
    r = ApiGetEpisodeAgentLogsRequest()
    r.episode_id = episode_id
    r.agent_index = agent_index
    return r


def _http_call_with_retry(
    label: str,
    fn,
    *,
    max_attempts: int = 16,
) -> requests.Response:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except requests.exceptions.HTTPError as e:
            last_exc = e
            r = e.response
            code = r.status_code if r is not None else None
            if code in (429, 503) and attempt < max_attempts - 1:
                wait = _backoff_seconds(attempt, r)
                print(f"    {label}: {code} Too Many Requests / backoff — sleeping {wait:.1f}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_ref", type=int, help='Submission "ref" (numeric id)')
    parser.add_argument(
        "logs_subdir",
        nargs="?",
        default=None,
        help="Folder under repo logs/ (default: submission_<ref>)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.35,
        metavar="SEC",
        help="Sleep after each episode's downloads (default: 0.35)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-download even when target files already exist and are non-empty",
    )
    args = parser.parse_args()
    sub_ref = args.submission_ref
    name = args.logs_subdir or f"submission_{sub_ref}"
    resume = not args.no_resume
    pause = max(0.0, args.pause_seconds)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    out = os.path.join(repo_root, "logs", name)
    replays_dir = os.path.join(out, "replays")
    agent_logs_dir = os.path.join(out, "agent_logs")

    api = KaggleApi()
    api.authenticate()

    episodes = api.competition_list_episodes(sub_ref) or []
    if not episodes:
        print("No episodes for this submission.", file=sys.stderr)
        sys.exit(1)

    print(f"Writing to {out}")
    print(f"Episodes: {len(episodes)}")

    with api.build_kaggle_client() as kaggle:
        for ep in episodes:
            ep_id = ep.id
            replay_path = os.path.join(replays_dir, f"episode-{ep_id}-replay.json")
            if resume and os.path.isfile(replay_path) and os.path.getsize(replay_path) > 0:
                print(f"  replay episode {ep_id} (skip, file exists)")
            else:
                print(f"  replay episode {ep_id}")
                req = ApiGetEpisodeReplayRequest()
                req.episode_id = ep_id
                replay_resp = _http_call_with_retry(
                    "GetEpisodeReplay",
                    lambda: kaggle.competitions.competition_api_client.get_episode_replay(req),
                )
                _stream_response_to_file(replay_resp, replay_path)

            for i in range(4):
                log_path = os.path.join(agent_logs_dir, f"episode-{ep_id}-agent-{i}-logs.json")
                if resume and os.path.isfile(log_path) and os.path.getsize(log_path) > 0:
                    continue
                try:
                    log_resp = _http_call_with_retry(
                        "GetEpisodeAgentLogs",
                        lambda i=i: kaggle.competitions.competition_api_client.get_episode_agent_logs(
                            _agent_logs_request(ep_id, i)
                        ),
                    )
                except requests.exceptions.RequestException:
                    continue
                _stream_response_to_file(log_resp, log_path)
                print(f"    logs agent {i}")

            if pause:
                time.sleep(pause)

    print("Done.")


if __name__ == "__main__":
    main()
