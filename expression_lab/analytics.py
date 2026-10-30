"""Bounded, in-memory track timelines and standalone HTML/JSON session reports."""

import html
import json
from collections import Counter, deque
from pathlib import Path

from .labels import RAW_LABELS


class SessionAnalytics:
    def __init__(self, history_size=240):
        self.history_size = history_size
        self.tracks: dict = {}
        self.frames = 0
        self.first_time = None
        self.last_time = 0.0
        self.events: list = []

    def update(self, faces, timestamp, events=()):
        self.frames += 1
        if self.first_time is None:
            self.first_time = timestamp
        self.last_time = timestamp
        self.events.extend(events)
        self.events = self.events[-1000:]
        present = {face.track_id for face in faces}
        for track_id, track in self.tracks.items():
            if track_id not in present:
                track["previous"] = None
        for face in faces:
            track = self.tracks.setdefault(
                face.track_id,
                {
                    "counts": Counter(),
                    "transitions": Counter(),
                    "history": deque(maxlen=self.history_size),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "previous": None,
                    "confidence_total": 0.0,
                    "fresh_samples": 0,
                    "quality_rejections": 0,
                },
            )
            track["last_seen"] = timestamp
            track["history"].append((timestamp, face.label, face.confidence))
            if face.label == "uncertain" or face.uncertainty_reason:
                track["previous"] = None
            if not face.fresh_prediction:
                continue
            track["fresh_samples"] += 1
            track["counts"][face.label] += 1
            track["confidence_total"] += face.confidence
            if face.uncertainty_reason == "image_quality":
                track["quality_rejections"] += 1
            previous = track["previous"]
            if previous and previous != face.label and face.label != "uncertain":
                track["transitions"][f"{previous} -> {face.label}"] += 1
            track["previous"] = face.label if face.label != "uncertain" else None

    def summary(self) -> dict:
        tracks = []
        for track_id, data in self.tracks.items():
            confident = {key: value for key, value in data["counts"].items() if key != "uncertain"}
            dominant = max(confident, key=lambda key: confident[key]) if confident else "uncertain"
            tracks.append(
                {
                    "track_id": track_id,
                    "first_seen": data["first_seen"],
                    "last_seen": data["last_seen"],
                    "dominant_expression": dominant,
                    "label_counts": dict(data["counts"]),
                    "transition_counts": dict(data["transitions"]),
                    "mean_confidence": data["confidence_total"] / max(1, data["fresh_samples"]),
                    "fresh_samples": data["fresh_samples"],
                    "quality_rejections": data["quality_rejections"],
                    "timeline": [
                        {"time": t, "label": label, "confidence": confidence}
                        for t, label, confidence in data["history"]
                    ],
                }
            )
        return {
            "schema_version": 1,
            "frames": self.frames,
            "duration_seconds": self.last_time - (self.first_time or 0),
            "timeline_limit_per_track": self.history_size,
            "tracks": tracks,
            "events": self.events,
            "limitations": "Temporary spatial tracks are not identities. Timelines retain the latest samples; counts cover fresh predictions. Scores estimate visible expressions, not feelings.",
        }

    def write_report(self, path: Path):
        summary = self.summary()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".json":
            with path.open("x", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, allow_nan=False)
            return
        if path.suffix.lower() != ".html":
            raise ValueError("Reports must be .html or .json")
        colors = (
            "#91b4d8",
            "#75d689",
            "#efd075",
            "#679edd",
            "#ed8387",
            "#9bbd76",
            "#ca9bdc",
            "#dca18d",
            "#8c929e",
        )
        palette = dict(zip((*RAW_LABELS, "uncertain"), colors))
        sections = []
        for track in summary["tracks"]:
            counts = track["label_counts"]
            maximum = max(counts.values(), default=1)
            bars = []
            for index, label in enumerate((*RAW_LABELS, "uncertain")):
                value = counts.get(label, 0)
                y = index * 27
                bars.append(
                    f'<text x="0" y="{y + 15}">{label}</text><rect x="100" y="{y}" width="{value / maximum * 350:.1f}" height="19" fill="{palette[label]}"/><text x="465" y="{y + 15}">{value}</text>'
                )
            timeline = track["timeline"]
            width = 700 / max(1, len(timeline))
            segments = "".join(
                f'<rect x="{i * width:.2f}" y="0" width="{width + 0.1:.2f}" height="34" fill="{palette[item["label"]]}"><title>{item["time"]:.2f}s: {item["label"]} {item["confidence"]:.0%}</title></rect>'
                for i, item in enumerate(timeline)
            )
            transitions = html.escape(json.dumps(track["transition_counts"]))
            sections.append(
                f'<section><h2>Track #{track["track_id"]}</h2><p>Dominant: <b>{track["dominant_expression"]}</b> · {track["fresh_samples"]} fresh predictions · {track["quality_rejections"]} quality rejections</p><svg viewBox="0 0 520 250" role="img" aria-label="Expression counts">{"".join(bars)}</svg><h3>Recent timeline (one segment per processed frame)</h3><svg viewBox="0 0 700 36" role="img" aria-label="Recent expression timeline">{segments}</svg><p>Transitions: {transitions}</p></section>'
            )
        body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Expression Lab session report</title><style>body{{font:16px system-ui;background:#14171c;color:#e6e9ee;max-width:980px;margin:40px auto;padding:0 24px}}section{{background:#20252d;padding:24px;margin:24px 0;border-radius:12px}}svg{{width:100%;max-width:700px}}svg text{{fill:#e6e9ee;font:14px system-ui}}p{{line-height:1.6}}h1{{font-size:36px}}small{{color:#b5becb}}</style><h1>Expression Lab / Session report</h1><p>{summary["frames"]} frames · {summary["duration_seconds"]:.2f} seconds · {len(summary["tracks"])} temporary tracks</p><small>{html.escape(summary["limitations"])}</small>{"".join(sections) or "<p>No faces detected.</p>"}<h2>Events</h2><pre>{html.escape(json.dumps(summary["events"], indent=2))}</pre></html>"""
        with path.open("x", encoding="utf-8") as handle:
            handle.write(body)
