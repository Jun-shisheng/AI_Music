"""Independent SMF reader tests the emitted MIDI, not just the event builder."""
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import wave

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.reference_studio.controlled_arrangement import prepare_controlled, sha, save


CHORDS = ["D#m", "B", "F#", "C#"]*2


def fixture(tmp_path):
    vocal = tmp_path/"comp.wav"
    with wave.open(str(vocal), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(24000)
        stream.writeframes(b"\0\0"*535824)
    metadata = []
    for version in (2, 3):
        path = tmp_path/f"target_v{version}.json"
        save(path, [{"index": f"v{version}", "duration": "22.325587", "time": [0, 22326],
                     "note_pitch": "66", "note_type": "2", "alignment_status": "approximate_alignment", "human_reviewed": False}])
        metadata.append({"path": str(path), "sha256": sha(path)})
    source = tmp_path/"source.abc"
    source.write_text('X:1\nM:4/4\nL:1/32\nQ:1/4=86\nK:D#m\n% chorus\n' +
                      ''.join('V: Vocal\n'+''.join(f'"{c}"F32|' for c in group)+'\nV: Ins\nZ4|\n' for group in (CHORDS[:4], CHORDS[4:])), encoding="utf-8")
    manifest = tmp_path/"vocal_manifest.json"
    save(manifest, {"status": "awaiting_listening", "kind": "phrase_comp",
                    "outputs": {"vocals": {"path": str(vocal), "sha256": sha(vocal)}},
                    "score_provenance": {"early": metadata[1], "late": metadata[0]}})
    config = tmp_path/"config.json"
    settings = {"schema": 1, "tempo_bpm": 86, "meter": "4/4", "bars": 8, "key": "D#m", "chords": CHORDS,
                "harmony_source": {"path": str(source), "sha256": sha(source), "section": "chorus", "occurrence": 1},
                "vocal_manifest": {"path": str(manifest), "sha256": sha(manifest), "output": "vocals"},
                "target_metadata": metadata}
    save(config, settings)
    return config, vocal, settings


def read_vlq(data, position):
    value = 0
    while True:
        byte = data[position]; position += 1
        value = value*128+(byte & 127)
        if byte < 128:
            return value, position


def read_smf(path):
    data = path.read_bytes()
    assert data[:4] == b"MThd"
    header_length, fmt, count, ppq = struct.unpack(">IHHH", data[4:14])
    assert (header_length, fmt, count, ppq) == (6, 1, 5, 480)
    offset, tracks = 14, []
    for _ in range(count):
        assert data[offset:offset+4] == b"MTrk"
        size = struct.unpack(">I", data[offset+4:offset+8])[0]
        end, cursor, tick, result = offset+8+size, offset+8, 0, []
        while cursor < end:
            delta, cursor = read_vlq(data, cursor); tick += delta
            status = data[cursor]; cursor += 1
            if status == 0xff:
                kind = data[cursor]; cursor += 1
                length, cursor = read_vlq(data, cursor)
                result.append((tick, "meta", kind, data[cursor:cursor+length]))
                cursor += length
            else:
                assert status >= 0x80, "Writer unexpectedly emitted running status"
                length = 1 if status & 0xf0 in (0xc0, 0xd0) else 2
                result.append((tick, "channel", status, data[cursor:cursor+length]))
                cursor += length
        assert cursor == end
        tracks.append(result); offset = end
    assert offset == len(data)
    return tracks


def test_standard_midi_has_exact_grid_channels_note_offs_and_json_events(tmp_path):
    config, vocal, _ = fixture(tmp_path)
    result = prepare_controlled(config, vocal, tmp_path/"out")
    assert result["kind"] == "controlled_arrangement_prepare"
    assert result["status"] == "prepared_not_rendered" and not result["rendering"]["mix_available"]
    clock = result["clock"]
    assert Fraction(clock["duration_exact_fraction"]) == Fraction(960, 43)
    assert clock["end_tick"] == 15360 and abs(clock["midi_tempo_rounding_error_seconds"]) < .00002
    for variant in ("sparse", "expanded"):
        records = result["variants"][variant]
        midi = tmp_path/"out"/records["midi"]["path"]
        assert sha(midi) == records["midi"]["sha256"]
        events = json.loads((tmp_path/"out"/records["events"]["path"]).read_text())
        tracks = read_smf(midi)
        assert int.from_bytes(next(x[3] for x in tracks[0] if x[1:3] == ("meta", 0x51)), "big") == 697674
        assert [x for x in tracks[0] if x[1:3] == ("meta", 6)][-1][3] == b"END"
        assert next(x[3] for x in tracks[0] if x[1:3] == ("meta", 0x59)) == bytes([6, 1])
        for role, midi_track in zip(("drums", "bass", "chords", "strings"), tracks[1:]):
            spec = events["tracks"][role]
            assert midi_track[-1][:3] == (15360, "meta", 0x2f)
            active, notes = {}, []
            for tick, kind, status, payload in midi_track:
                if kind != "channel":
                    continue
                assert status & 15 == spec["channel_zero_based"]
                if status & 0xf0 == 0x90:
                    assert payload[0] not in active
                    active[payload[0]] = (tick, payload[1])
                elif status & 0xf0 == 0x80:
                    start, velocity = active.pop(payload[0])
                    notes.append((start, tick, payload[0], velocity))
            assert not active
            expected = [(n["start_tick"], n["end_tick"], n["note"], n["velocity"]) for n in spec["notes"]]
            assert sorted(notes) == sorted(expected)
            assert all(0 <= start < end <= 15360 for start, end, _, _ in notes)
    expanded = json.loads((tmp_path/"out/expanded/events.json").read_text())
    # C# major contains E# (pitch class 5), not E natural.
    assert {n["note"] % 12 for n in expanded["tracks"]["chords"]["notes"] if n["start_tick"] == 3*1920} == {1, 5, 8}
    sparse = json.loads((tmp_path/"out/sparse/events.json").read_text())
    assert sparse["tracks"]["strings"]["first_note_tick"] == 6*1920
    assert expanded["tracks"]["strings"]["first_note_tick"] == 2*1920


def test_two_variants_copy_one_vocal_bit_exact_and_never_render_instruments(tmp_path):
    config, vocal, _ = fixture(tmp_path)
    before = sha(vocal)
    result = prepare_controlled(config, vocal, tmp_path/"out")
    refs = [v["vocal_reference"] for v in result["variants"].values()]
    assert refs[0] == refs[1]
    assert refs[0]["sha256"] == before == sha(vocal) == sha(tmp_path/"out"/refs[0]["path"])
    assert not refs[0]["regenerated"]
    assert list((tmp_path/"out").rglob("*.wav")) == [tmp_path/"out/source/fixed_vocal.wav"]
    assert not result["rendering"]["performed"] and not result["rendering"]["gpu_used"]


def test_nonempty_directory_is_never_overwritten(tmp_path):
    config, vocal, _ = fixture(tmp_path)
    out = tmp_path/"out"; out.mkdir()
    sentinel = out/"prior.mid"; sentinel.write_bytes(b"prior")
    with pytest.raises(ValueError, match="strictly empty"):
        prepare_controlled(config, vocal, out)
    assert sentinel.read_bytes() == b"prior" and list(out.iterdir()) == [sentinel]


def test_config_changed_after_parsing_cannot_be_archived_as_validated(tmp_path, monkeypatch):
    from tools.reference_studio import controlled_arrangement as controlled
    config, vocal, _ = fixture(tmp_path)
    original = controlled.validate_inputs

    def validate_then_change(*args):
        result = original(*args)
        # Even a semantically equivalent edit has different bytes from the parsed input.
        with config.open("ab") as source:
            source.write(b"\n")
        return result

    monkeypatch.setattr(controlled, "validate_inputs", validate_then_change)
    with pytest.raises(ValueError, match="changed before copying"):
        controlled.prepare_controlled(config, vocal, tmp_path/"out")
    assert not (tmp_path/"out/manifest.json").exists()
    assert not (tmp_path/"out/source/arrangement_config.json").exists()


@pytest.mark.parametrize("change", ["chord", "tempo", "vocal_hash", "metadata", "regenerate"])
def test_invalid_provenance_or_voice_change_is_rejected_before_outputs(tmp_path, change):
    config, vocal, settings = fixture(tmp_path)
    if change == "chord":
        settings["chords"][0] = "C#m"
    elif change == "tempo":
        settings["tempo_bpm"] = 120
    elif change == "vocal_hash":
        with vocal.open("ab") as output:
            output.write(b"mutation")
    elif change == "metadata":
        settings["target_metadata"] = settings["target_metadata"][:1]
    else:
        settings["regenerate_voice"] = True
    save(config, settings)
    with pytest.raises(ValueError):
        prepare_controlled(config, vocal, tmp_path/"out")
    assert not (tmp_path/"out").exists()
