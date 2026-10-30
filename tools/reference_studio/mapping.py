"""Propose destination mappings from measured contrast without relabeling source.

The proposal identifies a pair of adjacent reference intervals with a strong
energy/transient change. It does not decide that those intervals are a verse
and a chorus, and does not identify their instruments.
"""
from __future__ import annotations

import copy
import math
import statistics


def propose_mapping(blueprint: dict, target_sections=("verse", "chorus"),
                    source_sections=None, replace=False) -> dict:
    if len(target_sections) != 2 or len(set(target_sections)) != 2:
        raise ValueError("Provide two distinct destination plan sections")
    sections = blueprint.get("sections", [])
    if len(sections) < 2:
        raise ValueError("At least two measured reference intervals are required")
    if any(s.get("target", {}).get("plan_section") for s in sections) and not replace:
        raise ValueError("Blueprint already has explicit destination mappings; edit manually or use --replace-mapping")
    measurements = []
    for index, section in enumerate(sections):
        mix = section.get("measured_mix", {})
        rms = mix.get("rms_dbfs")
        transient = mix.get("mix_transient_events_per_s")
        if not isinstance(rms, (int, float)) or not math.isfinite(rms):
            raise ValueError(f"Reference section {section.get('name')} lacks measured rms_dbfs")
        if not isinstance(transient, (int, float)) or not math.isfinite(transient):
            raise ValueError(f"Reference section {section.get('name')} lacks measured transient rate")
        measurements.append({"index": index, "name": section.get("name"), "rms_dbfs": rms,
                             "mix_transient_events_per_s": transient})
    rms_scale = max(statistics.pstdev(m["rms_dbfs"] for m in measurements), 1.0)
    transient_scale = max(statistics.pstdev(m["mix_transient_events_per_s"] for m in measurements), .2)
    candidates = []
    if source_sections:
        if len(source_sections) != 2:
            raise ValueError("--source-sections requires two source interval names")
        indices = []
        for name in source_sections:
            matches = [m["index"] for m in measurements if m["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"Source interval {name!r} is missing or ambiguous")
            indices.append(matches[0])
        if indices[1] <= indices[0]:
            raise ValueError("Selected source intervals must remain in time order")
        pairs = [indices]
    else:
        pairs = [(i, i + 1) for i in range(len(measurements) - 1)]
    for first, second in pairs:
        a, b = measurements[first], measurements[second]
        energy_delta = b["rms_dbfs"] - a["rms_dbfs"]
        transient_delta = b["mix_transient_events_per_s"] - a["mix_transient_events_per_s"]
        # A descriptive rank used only to propose which interval pair to try.
        # It is not a quality measure or a claim of causal instrumentation.
        contrast = .65 * energy_delta / rms_scale + .35 * transient_delta / transient_scale
        candidates.append({"source_indices": [first, second],
                           "energy_delta_db": round(energy_delta, 5),
                           "mix_transient_rate_delta_per_s": round(transient_delta, 5),
                           "descriptor_contrast": round(contrast, 5)})
    chosen = max(candidates, key=lambda row: abs(row["descriptor_contrast"]))
    out = copy.deepcopy(blueprint)
    if replace:
        for section in out["sections"]:
            section.setdefault("target", {}).pop("plan_section", None)
    if abs(chosen["descriptor_contrast"]) < .1:
        out["mapping_proposal"] = {
            "status": "no_measurable_contrast_no_mapping_proposed", "selected_transition": chosen,
            "notes": "Intervals have too little measured contrast; choose and annotate source intervals manually."}
        return out
    first_index, second_index = chosen["source_indices"]
    patterns = ("sparse", "full") if chosen["descriptor_contrast"] > 0 else ("full", "sparse")
    for index, destination, pattern in zip([first_index, second_index], target_sections, patterns):
        target = out["sections"][index].setdefault("target", {})
        target.update({"plan_section": destination, "ins_pattern": pattern,
                       "proposal_status": "provisional_source_role_unknown",
                       "derivation": {"method": "relative_energy_and_mix_transient_contrast",
                                      "measured_reference_interval": measurements[index],
                                      "selected_transition": chosen},
                       "destination_slot_note": "Destination plan section only; no source verse/chorus identity was inferred."})
    out["mapping_proposal"] = {
        "status": "provisional_source_role_unknown", "method": "strongest_adjacent_energy_and_mix_transient_contrast",
        "source_selection": "explicit_source_names" if source_sections else "automatic_adjacent_interval_pair",
        "selected_transition": chosen, "candidate_count": len(candidates),
        "destination_plan_sections": list(target_sections),
        "source_names_preserved": True, "human_listening_reviewed": False,
        "warning": "Energy/transient activity cannot identify source instruments, section roles or musical quality. Inspect the proposed direction before rendering."}
    return out
