#!/usr/bin/env python3
"""
Extract rich-mode dance choreography from session.log and save to ctml_repo.

Usage:
    python scripts/extract_choreography.py [--log session.log] [--dry-run]

Rules:
- Only runs when REACHY_MINI_PERFORMANCE_MODE=rich (in .env)
- Finds play_music commands without a non-(-1) duration parameter
- Looks up bvid from music_cache/index.json via query string
- Finds the following long AI CTML block (dance choreography)
- Strips ANSI codes and exception/traceback noise
- Validates tag pairing, picks longest valid block per song
- Skips songs already saved in ctml_repo
- Saves to .workspace/runtime/ctml_repo/song_{bvid}.ctml
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # project root
ENV_FILE = REPO_ROOT / ".env"
WORKSPACE = REPO_ROOT / "src" / "moss_in_reachy_mini" / ".workspace" / "runtime"
MUSIC_CACHE_INDEX = WORKSPACE / "music_cache" / "index.json"
CTML_REPO_DIR = WORKSPACE / "ctml_repo"
DEFAULT_LOG = WORKSPACE / "logs" / "terminal.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKABCDEFGHJKSTfhilmnprsu]")

# Minimum tag count to consider a CTML block "long enough"
MIN_TAG_COUNT = 10


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            env[k.strip()] = v
    return env


def is_rich_mode(env: dict[str, str]) -> bool:
    mode = os.environ.get("REACHY_MINI_PERFORMANCE_MODE") or env.get("REACHY_MINI_PERFORMANCE_MODE", "")
    return mode.lower() == "rich"


def load_music_index(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_bvid(query: str, music_index: dict) -> str | None:
    """Look up bvid from cache index by query key or by title/artist substring match."""
    # exact key match
    if query in music_index:
        entry = music_index[query]
        return entry.get("bvid")

    # fuzzy: query words appear in title or artist
    query_lower = query.lower()
    best: str | None = None
    for key, entry in music_index.items():
        bvid = entry.get("bvid")
        if not bvid:
            continue
        title = entry.get("title", "").lower()
        artist = entry.get("artist", "").lower()
        combined = f"{artist} {title}"
        # check all query tokens appear somewhere
        tokens = query_lower.split()
        if all(t in combined or t in key.lower() for t in tokens):
            best = bvid
            break
    return best


# ---------------------------------------------------------------------------
# CTML validation
# ---------------------------------------------------------------------------
# Tags that are self-closing (no need for matching close tag)
VOID_TAGS_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_:]*)[^>]*/\s*>")
OPEN_TAG_RE = re.compile(r"<([a-zA-Z_][a-zA-Z0-9_:]*)(?:\s[^>]*)?>")
CLOSE_TAG_RE = re.compile(r"</([a-zA-Z_][a-zA-Z0-9_:]*)>")


def score_ctml(ctml: str) -> tuple[int, int]:
    """Return (tag_count, unique_tag_types) — higher is better."""
    tags = re.findall(r"<([a-zA-Z_][a-zA-Z0-9_:]*)", ctml)
    return len(tags), len(set(tags))


def clean_ctml(raw: str) -> str:
    """
    Remove noise injected into stdout mid-stream and strip ANSI codes.
    Noise appears inside CTML tags (splitting attribute values or tag names),
    so we use regex replacement on the raw string before any line processing.
    """
    # [MEM] monitor lines may appear mid-attribute, e.g. <sleep duration[MEM]...\n="0.48"/>
    raw = re.sub(r"\[MEM\][^\n]*\n?", "", raw)

    # "Task exception was never retrieved" blocks may appear mid-tag-name,
    # e.g. <jetarm:Task exception...\nTraceback...\nKeyError: '...'\nmotion name="..."/>
    # Remove from "Task exception" through the terminal XxxError/XxxException line.
    raw = re.sub(
        r"Task exception was never retrieved.*?[A-Za-z]+(?:Error|Exception):[^\n]*\n?",
        "",
        raw,
        flags=re.DOTALL,
    )

    # Standalone Traceback blocks (not preceded by "Task exception")
    raw = re.sub(
        r"Traceback \(most recent call last\):.*?[A-Za-z]+(?:Error|Exception):[^\n]*\n?",
        "",
        raw,
        flags=re.DOTALL,
    )

    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Single-line exception mentions not caught above
        if re.match(r"^[A-Za-z]+Error:", stripped) or re.match(r"^[A-Za-z]+Exception:", stripped):
            continue
        # Stack frame lines
        if re.match(r"^\s+(File |During handling|The above exception)", line):
            continue
        cleaned.append(strip_ansi(line))
    return "".join(cleaned)


def is_valid_ctml(ctml: str) -> bool:
    """
    Validate that all non-self-closing open tags have matching close tags.
    Uses a simple stack approach — colons in tag names are treated as plain chars
    so ET doesn't interpret them as namespace prefixes.
    """
    # Replace namespace-style prefixes (foo:bar) with flat names (foo__bar)
    # so ElementTree doesn't reject them as undefined namespace prefixes.
    normalized = re.sub(r"<(/?)([a-zA-Z_][a-zA-Z0-9_]*):", r"<\1\2__", ctml)
    wrapped = f"<root>{normalized}</root>"
    try:
        ET.fromstring(wrapped)
        return True
    except ET.ParseError:
        return False


def strip_to_valid_prefix(ctml: str) -> str:
    """
    Progressively trim trailing incomplete tags from ctml until it is valid XML.
    Returns the longest valid prefix, or empty string if nothing validates.
    """
    # Find all tag boundaries (end positions of complete tags)
    tag_ends = [m.end() for m in re.finditer(r"<[^>]+>", ctml)]
    for i in range(len(tag_ends) - 1, -1, -1):
        candidate = ctml[: tag_ends[i]]
        if is_valid_ctml(candidate):
            return candidate
    return ""


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
AI_BLOCK_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] AI:\s*$")

# Matches: <reachy_mini:play_music query="..." .../>
PLAY_MUSIC_RE = re.compile(r"<reachy_mini:play_music\s([^>]*)/>")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_play_music_attrs(tag_body: str) -> dict[str, str]:
    return {k: v for k, v in ATTR_RE.findall(tag_body)}


def is_truncated(attrs: dict[str, str]) -> bool:
    """Return True if the play_music has a real duration limit (not absent and not '-1')."""
    duration = attrs.get("duration")
    if duration is None:
        return False
    return duration != "-1"


def extract_dance_ctml_from_line(line: str) -> str:
    """
    From a raw AI output line, extract only the dance/motion CTML,
    stripping <say> blocks and other non-motion content.
    Returns empty string if this looks like a loop-mode block (save_ctml name="music_loop").
    """
    # Loop mode outputs <save_ctml name="music_loop"> — skip entirely
    # Rich mode save_performance outputs <save_ctml name="song_..."> — keep inner content
    if re.search(r'<save_ctml\s[^>]*name="music_loop"', line):
        return ""
    line = clean_ctml(line)
    # Remove <say>...</say> blocks
    line = re.sub(r"<say>.*?</say>", "", line, flags=re.DOTALL)
    # Remove play_music, stop_music, etc. (reachy_mini commands that aren't motion)
    line = re.sub(r"<reachy_mini:(?:play_music|stop_music|pause_music|resume_music|search_music)[^>]*/?>", "", line)
    # Remove <save_ctml ...>...</save_ctml> wrappers but keep inner content (e.g. save_performance)
    line = re.sub(r"<save_ctml[^>]*>(.*?)</save_ctml>", r"\1", line, flags=re.DOTALL)
    return line.strip()


def parse_log(log_path: Path) -> list[dict]:
    """
    Parse session.log and return a list of candidate records:
      {query, bvid_hint, ctml_candidates: [str]}

    Each record is anchored to a play_music command.
    CTML candidates are the AI blocks that follow (before the next play_music).
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    records: list[dict] = []
    current_record: dict | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect AI block header
        if AI_BLOCK_RE.match(line.strip()) or re.match(r"^\[\d{2}:\d{2}:\d{2}\] AI:\s*$", line):
            # Next non-empty line is the AI content
            i += 1
            content_parts: list[str] = []
            while i < len(lines):
                raw = lines[i]
                stripped = strip_ansi(raw).strip()
                if stripped.startswith("(first_token_cost:"):
                    break
                if stripped.startswith("[") and re.match(r"^\[\d{2}:\d{2}:\d{2}\]", stripped):
                    break
                if stripped.startswith(">"):
                    break
                content_parts.append(raw)
                i += 1
            content = "\n".join(content_parts)

            # Does this AI block contain a play_music command?
            pm_match = PLAY_MUSIC_RE.search(content)
            if pm_match:
                attrs = parse_play_music_attrs(pm_match.group(1))
                query = attrs.get("query", "")
                if query and not is_truncated(attrs):
                    current_record = {
                        "query": query,
                        "ctml_candidates": [],
                    }
                    records.append(current_record)
                else:
                    # truncated or no query — close current record
                    current_record = None
            elif current_record is not None:
                # This AI block might be the dance choreography
                dance_ctml = extract_dance_ctml_from_line(content)
                if dance_ctml:
                    current_record["ctml_candidates"].append(dance_ctml)
            continue

        i += 1

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract choreography from session.log")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Path to terminal.log")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be saved without writing")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    load_env(ENV_FILE)
    music_index = load_music_index(MUSIC_CACHE_INDEX)
    CTML_REPO_DIR.mkdir(parents=True, exist_ok=True)

    records = parse_log(log_path)
    print(f"Found {len(records)} play_music event(s) in log.")

    # Group all CTML candidates by bvid across the whole log
    bvid_to_info: dict[str, dict] = {}  # bvid -> {query, candidates: [str]}
    no_bvid: list[str] = []
    for rec in records:
        query = rec["query"]
        bvid = find_bvid(query, music_index)
        if not bvid:
            no_bvid.append(query)
            continue
        if bvid not in bvid_to_info:
            bvid_to_info[bvid] = {"query": query, "candidates": []}
        bvid_to_info[bvid]["candidates"].extend(rec["ctml_candidates"])

    for query in no_bvid:
        print(f"  [SKIP] '{query}' — no bvid found in music cache index")

    saved = 0
    skipped = 0

    for bvid, info in bvid_to_info.items():
        query = info["query"]
        candidates = info["candidates"]

        save_path = CTML_REPO_DIR / f"song_{bvid}.ctml"

        # Pick the best (longest + most diverse) CTML candidate across all log occurrences
        best_ctml = ""
        best_score: tuple[int, int] = (0, 0)
        for candidate in candidates:
            if is_valid_ctml(candidate):
                valid = candidate
            else:
                valid = strip_to_valid_prefix(candidate)
            if not valid:
                continue
            s = score_ctml(valid)
            if s > best_score:
                best_score = s
                best_ctml = valid

        tag_count, unique_types = best_score

        if not best_ctml or tag_count < MIN_TAG_COUNT:
            print(f"  [SKIP] '{query}' (bvid={bvid}) — no sufficiently long valid CTML found "
                  f"(best: {tag_count} tags, candidates: {len(candidates)})")
            skipped += 1
            continue

        # Compare against existing file (update only if new version is strictly better)
        if save_path.exists():
            existing = save_path.read_text(encoding="utf-8")
            existing_score = score_ctml(existing)
            if best_score <= existing_score:
                print(f"  [SKIP] '{query}' (bvid={bvid}) — existing is equally good or better "
                      f"(existing: {existing_score[0]} tags/{existing_score[1]} types, "
                      f"new: {tag_count} tags/{unique_types} types)")
                skipped += 1
                continue
            action = "UPDATE"
        else:
            action = "SAVE"

        formatted = re.sub(r"(>)(<)", r"\1\n\2", best_ctml).strip()
        print(f"  [{action}] '{query}' (bvid={bvid}) → {save_path.name}  "
              f"({tag_count} tags, {unique_types} types)")
        if not args.dry_run:
            save_path.write_text(formatted + "\n", encoding="utf-8")
        saved += 1

    print(f"\nDone. Saved: {saved}, Skipped: {skipped}.")
    if args.dry_run:
        print("(dry-run mode — nothing was written)")


if __name__ == "__main__":
    main()
