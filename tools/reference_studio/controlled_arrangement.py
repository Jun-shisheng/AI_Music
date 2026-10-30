"""Prepare deterministic MIDI arrangements around one immutable vocal WAV.

Standard-library only. Produces no instrument audio and performs no inference.
The native ABC is a cited harmony source, not an audio-alignment guarantee.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import struct
import wave

from . import plan

PPQ = 480
ROLES = ("drums", "bass", "chords", "strings")
CHANNELS = {"drums": 9, "bass": 0, "chords": 1, "strings": 2}
PROGRAMS = {"drums": None, "bass": 33, "chords": 0, "strings": 48}
PAN = {"drums": 64, "bass": 64, "chords": 54, "strings": 74}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_record(record: dict, parent: Path) -> tuple[Path, str]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = parent / path
    path = path.resolve()
    expected = record.get("sha256")
    if not expected or not path.is_file() or sha(path) != expected:
        raise ValueError(f"Source file missing or SHA mismatch: {path}")
    return path, expected


def chord_tones(symbol: str) -> tuple[int, tuple[int, int, int]]:
    match = re.fullmatch(r'([A-G])([#b]?)(m?)', symbol)
    if not match:
        raise ValueError(f"Only cited major/minor triads supported: {symbol!r}")
    root = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match[1]]
    root = (root + (1 if match[2] == "#" else -1 if match[2] == "b" else 0)) % 12
    return root, (0, 3 if match[3] else 4, 7)


def validate_harmony(config: dict, source: Path) -> None:
    abc = source.read_text(encoding="utf-8-sig")
    header, sections = plan.parse(abc)
    clock = plan.timing(header)
    wanted = config["harmony_source"].get("section", "chorus").lower()
    occurrence = config["harmony_source"].get("occurrence", 1)
    choices = [s for s in sections if plan.section_name(s.marker) == wanted]
    if type(occurrence) is not int or not 1 <= occurrence <= len(choices):
        raise ValueError("Cited source section occurrence does not exist")
    fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
              for line in header if ":" in line}
    if fields.get("M") != "4/4" or fields.get("K") != config["key"] or clock["quarter_bpm"] != config["tempo_bpm"]:
        raise ValueError("Source meter/key/tempo differs from explicit arrangement parameters")
    bars = [bar for chunk in choices[occurrence-1].chunks for bar in plan.bars(chunk.vocal, clock["units_per_bar"])]
    symbols = []
    for bar in bars:
        chords = plan.CHORD.findall(bar)
        if len(chords) != 1 or not bar.lstrip().startswith('"'):
            raise ValueError("Require one cited chord at each bar downbeat")
        symbols.append(chords[0])
    if symbols != config["chords"] or len(symbols) != config["bars"]:
        raise ValueError("Explicit bar chords differ from cited ABC; harmony must not be invented")


def validate_inputs(config_file: Path, vocal: Path):
    config_bytes = config_file.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes.decode("utf-8-sig"))
    allowed = {"schema", "tempo_bpm", "bars", "meter", "key", "chords", "harmony_source", "vocal_manifest", "target_metadata", "notes"}
    if set(config)-allowed:
        raise ValueError(f"Unsupported parameters (vocal regeneration/variant processing is not supported): {sorted(set(config)-allowed)}")
    tempo, bars = config.get("tempo_bpm"), config.get("bars")
    if type(tempo) is not int or not 20 <= tempo <= 300:
        raise ValueError("tempo_bpm must be an integer between 20 and 300")
    if type(bars) is not int or not 1 <= bars <= 64 or config.get("meter") != "4/4":
        raise ValueError("Use 1..64 bars and explicit meter 4/4")
    if config.get("key") != "D#m":
        raise ValueError("This bounded preparer currently supports the cited D#m score")
    if not isinstance(config.get("chords"), list) or len(config["chords"]) != bars:
        raise ValueError("Provide one chord per bar")
    for chord in config["chords"]:
        chord_tones(chord)
    parent = config_file.resolve().parent
    harmony, harmony_hash = source_record(config["harmony_source"], parent)
    validate_harmony(config, harmony)
    manifest_file, manifest_hash = source_record(config["vocal_manifest"], parent)
    manifest = load(manifest_file)
    if manifest.get("status") not in {"awaiting_listening", "technical_review_required", "complete"}:
        raise ValueError("Vocal source must be a completed, reviewable artifact")
    output = manifest.get("outputs", {}).get(config["vocal_manifest"].get("output", "vocals"), {})
    vocal = vocal.resolve()
    vocal_hash = sha(vocal)
    if not output.get("sha256") or vocal_hash != output["sha256"]:
        raise ValueError("Supplied vocal does not match the source manifest output SHA")
    with wave.open(str(vocal), "rb") as stream:
        info = {"frames": stream.getnframes(), "sample_rate": stream.getframerate(),
                "channels": stream.getnchannels(), "sample_width": stream.getsampwidth()}
    duration = Fraction(info["frames"], info["sample_rate"])
    planned = Fraction(bars * 4 * 60, tempo)
    if abs(duration-planned) > Fraction(2, 1000):
        raise ValueError("Vocal container differs from the grid by more than 2ms; do not stretch or regenerate it silently")
    allowed_metadata = {r["sha256"] for r in manifest.get("score_provenance", {}).values()}
    if manifest.get("inputs", {}).get("score_sha256"):
        allowed_metadata.add(manifest["inputs"]["score_sha256"])
    metadata = []
    for record in config.get("target_metadata", []):
        path, digest = source_record(record, parent)
        if digest not in allowed_metadata:
            raise ValueError("Target metadata is not cited by the actual vocal source manifest")
        entries = load(path)
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError("Require one continuous target metadata entry")
        entry = entries[0]
        total = sum((Fraction(s) for s in entry["duration"].split()), Fraction(0))
        if abs(total-planned) > Fraction(2, 1000) or entry["time"][0] != 0 or abs(Fraction(entry["time"][1], 1000)-duration) > Fraction(2, 1000):
            raise ValueError("Target metadata timeline differs from the vocal/grid")
        metadata.append((path, digest))
    if not metadata or (manifest.get("kind") == "phrase_comp" and {h for _, h in metadata} != allowed_metadata):
        raise ValueError("Cite the actual target metadata for every parent of this vocal comp")
    return config, {"config": (config_file.resolve(), config_hash),
                    "harmony": (harmony, harmony_hash), "vocal_manifest": (manifest_file, manifest_hash),
                    "target_metadata": metadata, "vocal": (vocal, vocal_hash)}, info


def make_events(chords: list[str], tempo: int, variant: str) -> dict:
    if variant not in {"sparse", "expanded"}:
        raise ValueError("Unknown arrangement variant")
    tracks = {role: {"channel_zero_based": CHANNELS[role], "gm_program_zero_based": PROGRAMS[role],
                     "pan_cc10": PAN[role], "notes": []} for role in ROLES}
    end_tick = len(chords)*4*PPQ

    def note(role, bar, beat, length, pitch, velocity):
        start = (Fraction(bar*4)+Fraction(beat))*PPQ
        stop = start+Fraction(length)*PPQ
        if start.denominator != 1 or stop.denominator != 1 or not 0 <= start < stop <= end_tick:
            raise ValueError("Event falls outside the exact MIDI time grid")
        tracks[role]["notes"].append({"start_tick": int(start), "end_tick": int(stop),
            "note": pitch, "velocity": velocity,
            "start_seconds": float(start/PPQ*Fraction(60, tempo)),
            "end_seconds": float(stop/PPQ*Fraction(60, tempo))})

    for bar, chord in enumerate(chords):
        root, intervals = chord_tones(chord)
        if variant == "sparse":
            note("drums", bar, 0, Fraction(1, 8), 36, 57)
            note("drums", bar, 2, Fraction(1, 8), 37, 49)
            note("bass", bar, 0, 2, 36+root, 62)
            for interval in (0, 7):
                note("chords", bar, 0, 2, 48+root+interval, 49)
            if bar >= max(0, len(chords)-2):
                for interval in (0, 7):
                    note("strings", bar, 0, 4, 60+root+interval, 30)
        else:
            for beat in (0, Fraction(3, 2), 2, Fraction(7, 2)):
                note("drums", bar, beat, Fraction(1, 8), 36, 76 if beat in (0, 2) else 64)
            for beat in (1, 3):
                note("drums", bar, beat, Fraction(1, 8), 38, 78)
            for half in range(8):
                note("drums", bar, Fraction(half, 2), Fraction(1, 8), 42, 52 if half%2 == 0 else 42)
            if bar in (0, 4):
                note("drums", bar, 0, Fraction(1, 4), 49, 66)
            for beat in range(4):
                note("bass", bar, beat, Fraction(7, 8), 36+root+(7 if beat%2 else 0), 72)
            for beat, length in ((0, 1), (Fraction(3, 2), Fraction(1, 2)), (Fraction(5, 2), 1)):
                for interval in intervals:
                    note("chords", bar, beat, length, 48+root+interval, 61)
            if bar >= 2:
                for interval in intervals:
                    note("strings", bar, 0, 4, 60+root+interval, 41)
    for role, track in tracks.items():
        track["notes"].sort(key=lambda e: (e["start_tick"], e["note"]))
        track["first_note_tick"] = min((n["start_tick"] for n in track["notes"]), default=None)
        track["last_note_off_tick"] = max((n["end_tick"] for n in track["notes"]), default=None)
        track["all_notes_and_sound_off_tick"] = end_tick
    return {"schema": 1, "variant": variant, "ticks_per_quarter": PPQ, "end_tick": end_tick,
            "tracks": tracks, "rendered": False}


def vlq(value: int) -> bytes:
    if not 0 <= value <= 0x0fffffff:
        raise ValueError("MIDI delta outside VLQ range")
    result = [value & 127]
    while value >> 7:
        value >>= 7
        result.insert(0, (value & 127)|128)
    return bytes(result)


def meta(kind: int, payload: bytes) -> bytes:
    return b"\xff"+bytes([kind])+vlq(len(payload))+payload


def track_chunk(events: list[tuple[int, int, bytes]], end_tick: int) -> bytes:
    events = sorted([*events, (end_tick, 100, meta(0x2f, b""))], key=lambda row: (row[0], row[1], row[2]))
    data, previous = bytearray(), 0
    for tick, _, payload in events:
        data.extend(vlq(tick-previous)); data.extend(payload); previous = tick
    return b"MTrk"+struct.pack(">I", len(data))+data


def midi_bytes(events: dict, tempo: int, chords: list[str]) -> bytes:
    end = events["end_tick"]
    micros = round(Fraction(60000000, tempo))
    conductor = [(0, -4, meta(3, b"Conductor")), (0, -3, meta(0x51, micros.to_bytes(3, "big"))),
                 (0, -2, meta(0x58, bytes([4, 2, 24, 8]))), (0, -1, meta(0x59, bytes([6, 1])))]
    for index, chord in enumerate(chords):
        conductor.append((index*4*PPQ, 0, meta(6, f"BAR {index+1:02d} {chord}".encode())))
    conductor.append((end, 0, meta(6, b"END")))
    tracks = [track_chunk(conductor, end)]
    for role in ROLES:
        data, channel = events["tracks"][role], CHANNELS[role]
        stream = [(0, -3, meta(3, role.encode())), (0, -1, bytes([0xb0|channel, 10, PAN[role]]))]
        if PROGRAMS[role] is not None:
            stream.append((0, -2, bytes([0xc0|channel, PROGRAMS[role]])))
        for note in data["notes"]:
            stream += [(note["start_tick"], 2, bytes([0x90|channel, note["note"], note["velocity"]])),
                       (note["end_tick"], 0, bytes([0x80|channel, note["note"], 0]))]
        stream += [(end, 1, bytes([0xb0|channel, 123, 0])), (end, 2, bytes([0xb0|channel, 120, 0]))]
        tracks.append(track_chunk(stream, end))
    return b"MThd"+struct.pack(">IHHH", 6, 1, len(tracks), PPQ)+b"".join(tracks)


def prepare_controlled(config: Path, vocal: Path, out_dir: Path) -> dict:
    """Validate cited sources, then write into a strictly empty output folder."""
    config, vocal, out_dir = config.resolve(), vocal.resolve(), out_dir.resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError("Output directory must be absent or strictly empty; choose a new version")
    settings, sources, info = validate_inputs(config, vocal)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        raise ValueError("Output directory became nonempty during source validation; no files were replaced")
    lock = out_dir/".prepare.lock"
    with lock.open("x", encoding="ascii") as handle:
        handle.write("preparing; no rendering\n")
    snapshots = {}

    def snapshot(source: Path, relative: str, expected=None):
        destination = out_dir/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha(source)
        if expected and before != expected:
            raise ValueError("Validated source changed before copying")
        shutil.copy2(source, destination)
        if sha(destination) != before:
            raise OSError("Source snapshot copy failed hash verification")
        record = {"path": relative, "sha256": before, "source": str(source)}
        snapshots[relative] = record
        return record

    snapshot(config, "source/arrangement_config.json", sources["config"][1])
    fixed = snapshot(vocal, "source/fixed_vocal.wav", sources["vocal"][1])
    snapshot(sources["harmony"][0], "source/harmony.abc", sources["harmony"][1])
    snapshot(sources["vocal_manifest"][0], "source/vocal_manifest.json", sources["vocal_manifest"][1])
    for index, (path, digest) in enumerate(sources["target_metadata"]):
        snapshot(path, f"source/target_metadata_{index+1}.json", digest)
    bars, tempo = settings["bars"], settings["tempo_bpm"]
    duration = Fraction(bars*4*60, tempo)
    micros = round(Fraction(60000000, tempo))
    clock = {"tempo_bpm": tempo, "meter": "4/4", "key": settings["key"], "bars": bars,
             "quarter_beats": bars*4, "ticks_per_quarter": PPQ, "start_tick": 0, "end_tick": bars*4*PPQ,
             "duration_seconds": float(duration), "duration_exact_fraction": str(duration),
             "midi_microseconds_per_quarter": micros,
             "midi_encoded_duration_seconds": bars*4*micros/1000000,
             "midi_tempo_rounding_error_seconds": float(Fraction(bars*4*micros, 1000000)-duration),
             "vocal_container_duration_seconds": info["frames"]/info["sample_rate"],
             "vocal_container_minus_grid_seconds": float(Fraction(info["frames"], info["sample_rate"])-duration)}
    markers = {"clock": clock, "bars": [{"bar": i+1, "chord": chord, "start_tick": i*4*PPQ,
                "end_tick": (i+1)*4*PPQ, "start_seconds": float(Fraction(i*4*60, tempo)),
                "end_seconds": float(Fraction((i+1)*4*60, tempo))} for i, chord in enumerate(settings["chords"])],
               "vocal_offset_tick": 0, "ending": "END marker and all-notes/all-sound-off at the last bar boundary; no generated audio has been cut"}
    save(out_dir/"markers.json", markers)
    variants = {}
    for variant in ("sparse", "expanded"):
        events = make_events(settings["chords"], tempo, variant)
        events.update(clock=clock, vocal_reference={"path": fixed["path"], "sha256": fixed["sha256"], "start_tick": 0, "regenerated": False})
        directory = out_dir/variant
        directory.mkdir()
        save(directory/"events.json", events)
        (directory/"arrangement.mid").write_bytes(midi_bytes(events, tempo, settings["chords"]))
        variants[variant] = {"midi": {"path": f"{variant}/arrangement.mid", "sha256": sha(directory/"arrangement.mid")},
                             "events": {"path": f"{variant}/events.json", "sha256": sha(directory/"events.json")},
                             "vocal_reference": dict(events["vocal_reference"]),
                             "track_note_counts": {role: len(track["notes"]) for role, track in events["tracks"].items()}}
    code = {str(path): sha(path) for path in (Path(__file__), Path(plan.__file__))}
    identity = hashlib.sha256(json.dumps({"parameters": settings, "sources": {k:v["sha256"] for k,v in snapshots.items()},
                                          "code": code}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = {"schema": 1, "kind": "controlled_arrangement_prepare", "status": "prepared_not_rendered",
        "identity": identity, "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "clock": clock, "parameters": settings, "code_sha256": code, "sources": snapshots,
        "vocal_reference": {"path": fixed["path"], "sha256": fixed["sha256"], "regenerated": False, "parent_modified": False,
                            "wave": info, "listening_approved": False},
        "variants": variants, "markers": {"path": "markers.json", "sha256": sha(out_dir/"markers.json")},
        "rendering": {"performed": False, "mix_available": False, "instrument_audio_available": False,
                      "dependency_installation_performed": False, "gpu_used": False},
        "limits": ["MIDI programs identify intended GM roles; no real-instrument timbre or mixing quality has been rendered or verified.",
                   "The exact score grid is distinct from the vocal container length and MIDI integer-tempo rounding.",
                   "Harmony comes from cited symbolic ABC, not a verified transcription of the old generated backing.",
                   "Both variants reuse the same fixed vocal bytes. MIDI has no vocal note track and triggers no voice inference.",
                   "MIDI note-offs/end commands are authored controls; a future renderer must verify release tails and actual audio alignment.",
                   "Target SVS metadata and singer identity remain pending human listening; preparation does not approve lyrics or timbre."]}
    save(out_dir/"manifest.json", manifest)
    lock.unlink()
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocal", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_controlled(args.config, args.vocal, args.out_dir)
        print(json.dumps({"kind": result["kind"], "status": result["status"], "identity": result["identity"],
                          "duration_seconds": result["clock"]["duration_seconds"], "variants": list(result["variants"]),
                          "rendering": result["rendering"]}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, wave.Error) as error:
        print(f"Controlled preparation failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
