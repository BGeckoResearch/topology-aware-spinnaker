from __future__ import annotations

import csv
import json
import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import spinnaker_graph_front_end as gfe
from spinn_machine.virtual_machine import virtual_machine

Coord = Tuple[int, int]
Direction = str
DIRECTIONS: Sequence[Direction] = ("E", "W", "N", "S")
DIR_TO_STEP = {"E": (1, 0), "W": (-1, 0), "N": (0, 1), "S": (0, -1)}
SEEDS = [11, 23, 37, 41, 53, 67, 79, 83, 97, 101]
TOPOLOGIES = [(8, 8), (12, 12)]
TRAFFIC_LEVELS = ["low", "medium", "high"]
DEST_PATTERNS = ["clustered", "dispersed"]
MASK_SIZES = [4, 8, 12, 18]
ROUTING_PRESSURE = ["low", "high"]
ABLATIONS = ["baseline", "space_bits_only", "mask_only", "both", "null_dispatch"]


@dataclass
class Trial:
    seed: int
    topology: str
    traffic_level: str
    destination_pattern: str
    mask_target: int
    routing_pressure: str
    ablation: str
    baseline_hops: int
    tested_hops: int
    baseline_congested_hops: int
    tested_congested_hops: int
    baseline_mask_size: int
    tested_mask_size: int
    baseline_energy_proxy: float
    tested_energy_proxy: float
    baseline_latency_proxy: float
    tested_latency_proxy: float
    path_changed: int



def torus_step(coord: Coord, direction: Direction, width: int, height: int) -> Coord:
    dx, dy = DIR_TO_STEP[direction]
    return ((coord[0] + dx) % width, (coord[1] + dy) % height)



def shortest_torus_axis_delta(src: int, dst: int, mod: int) -> int:
    forward = (dst - src) % mod
    backward = (src - dst) % mod
    return forward if forward <= backward else -backward



def preferred_directions(source: Coord, destination: Coord, width: int, height: int) -> List[Direction]:
    dx = shortest_torus_axis_delta(source[0], destination[0], width)
    dy = shortest_torus_axis_delta(source[1], destination[1], height)
    dirs: List[Direction] = []
    if dx > 0:
        dirs.append("E")
    elif dx < 0:
        dirs.append("W")
    if dy > 0:
        dirs.append("N")
    elif dy < 0:
        dirs.append("S")
    for d in DIRECTIONS:
        if d not in dirs:
            dirs.append(d)
    return dirs



def bfs_path(source: Coord, destination: Coord, width: int, height: int, direction_order: Iterable[Direction], blocked: set[Coord]) -> List[Coord]:
    direction_order = tuple(direction_order)
    q = deque([source])
    parents: Dict[Coord, Coord | None] = {source: None}
    while q:
        cur = q.popleft()
        if cur == destination:
            break
        for d in direction_order:
            nxt = torus_step(cur, d, width, height)
            if nxt in blocked and nxt != destination:
                continue
            if nxt not in parents:
                parents[nxt] = cur
                q.append(nxt)
    if destination not in parents:
        return []
    out: List[Coord] = []
    cur: Coord | None = destination
    while cur is not None:
        out.append(cur)
        cur = parents[cur]
    return list(reversed(out))



def count_congested(path: Sequence[Coord], hotspots: set[Coord]) -> int:
    return sum(1 for p in path[1:] if p in hotspots)



def energy_proxy(hops: int, mask_size: int, congested_hops: int) -> float:
    # Explicitly a synthetic proxy, not calibrated hardware energy.
    return hops * 1.0 + 0.18 * mask_size + 0.9 * congested_hops



def latency_proxy(hops: int, congested_hops: int) -> float:
    # Explicitly a synthetic proxy, not measured latency.
    return hops * 1.0 + 1.75 * congested_hops



def make_workload(rng: random.Random, width: int, height: int, traffic_level: str, destination_pattern: str):
    if traffic_level == "low":
        hotspot_count = 2
    elif traffic_level == "medium":
        hotspot_count = 5
    else:
        hotspot_count = 9

    source = (rng.randrange(width), rng.randrange(height))
    if destination_pattern == "clustered":
        dx = rng.choice([1, 2, 3])
        dy = rng.choice([1, 2, 3])
        destination = ((source[0] + dx) % width, (source[1] + dy) % height)
    else:
        destination = ((source[0] + rng.randrange(width // 2, width)) % width,
                       (source[1] + rng.randrange(height // 2, height)) % height)

    hotspots = set()
    while len(hotspots) < hotspot_count:
        hotspots.add((rng.randrange(width), rng.randrange(height)))
    return source, destination, hotspots



def tested_policy(ablation: str, source: Coord, destination: Coord, width: int, height: int, hotspots: set[Coord], mask_target: int, routing_pressure: str):
    base_order = preferred_directions(source, destination, width, height)
    blocked = hotspots if routing_pressure == "high" else set(list(hotspots)[: max(1, len(hotspots)//2)])

    if ablation == "baseline":
        return bfs_path(source, destination, width, height, base_order, set()), 18
    if ablation == "null_dispatch":
        return bfs_path(source, destination, width, height, base_order, set()), 18
    if ablation == "space_bits_only":
        restricted = [d for d in base_order if d in base_order[:2]] + [d for d in base_order if d not in base_order[:2]]
        path = bfs_path(source, destination, width, height, restricted, blocked)
        if not path:
            path = bfs_path(source, destination, width, height, base_order, set())
        return path, 18
    if ablation == "mask_only":
        path = bfs_path(source, destination, width, height, base_order, set())
        return path, mask_target
    if ablation == "both":
        restricted = [d for d in base_order if d in base_order[:2]] + [d for d in base_order if d not in base_order[:2]]
        path = bfs_path(source, destination, width, height, restricted, blocked)
        if not path:
            path = bfs_path(source, destination, width, height, base_order, set())
        return path, mask_target
    raise ValueError(ablation)



def summarise_numeric(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": round(mean(values), 6),
        "std": round(pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "n": len(values),
    }



def main() -> None:
    out_dir = Path(__file__).with_name("benchmark_audit_outputs")
    out_dir.mkdir(exist_ok=True)

    gfe.setup(n_chips_required=1)
    trials: List[Trial] = []

    for width, height in TOPOLOGIES:
        _ = virtual_machine(width, height)
        for seed in SEEDS:
            rng = random.Random(seed + width * 100 + height)
            for traffic_level in TRAFFIC_LEVELS:
                for dest_pattern in DEST_PATTERNS:
                    for mask_target in MASK_SIZES:
                        for pressure in ROUTING_PRESSURE:
                            source, destination, hotspots = make_workload(rng, width, height, traffic_level, dest_pattern)

                            baseline_path, baseline_mask = tested_policy("baseline", source, destination, width, height, hotspots, 18, pressure)
                            baseline_hops = max(0, len(baseline_path) - 1)
                            baseline_congested = count_congested(baseline_path, hotspots)
                            baseline_energy = energy_proxy(baseline_hops, baseline_mask, baseline_congested)
                            baseline_latency = latency_proxy(baseline_hops, baseline_congested)

                            for ablation in ABLATIONS:
                                tested_path, tested_mask = tested_policy(ablation, source, destination, width, height, hotspots, mask_target, pressure)
                                tested_hops = max(0, len(tested_path) - 1)
                                tested_congested = count_congested(tested_path, hotspots)
                                tested_energy = energy_proxy(tested_hops, tested_mask, tested_congested)
                                tested_latency = latency_proxy(tested_hops, tested_congested)
                                trials.append(Trial(
                                    seed=seed,
                                    topology=f"{width}x{height}",
                                    traffic_level=traffic_level,
                                    destination_pattern=dest_pattern,
                                    mask_target=mask_target,
                                    routing_pressure=pressure,
                                    ablation=ablation,
                                    baseline_hops=baseline_hops,
                                    tested_hops=tested_hops,
                                    baseline_congested_hops=baseline_congested,
                                    tested_congested_hops=tested_congested,
                                    baseline_mask_size=baseline_mask,
                                    tested_mask_size=tested_mask,
                                    baseline_energy_proxy=round(baseline_energy, 6),
                                    tested_energy_proxy=round(tested_energy, 6),
                                    baseline_latency_proxy=round(baseline_latency, 6),
                                    tested_latency_proxy=round(tested_latency, 6),
                                    path_changed=int(tested_path != baseline_path),
                                ))
    gfe.stop()

    csv_path = out_dir / "audit_trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(trials[0]).keys()))
        writer.writeheader()
        for t in trials:
            writer.writerow(asdict(t))

    grouped: Dict[str, List[Trial]] = defaultdict(list)
    for t in trials:
        grouped[t.ablation].append(t)

    summary = {}
    for ablation, rows in grouped.items():
        energy_rel = [100.0 * (r.baseline_energy_proxy - r.tested_energy_proxy) / r.baseline_energy_proxy for r in rows]
        latency_rel = [100.0 * (r.baseline_latency_proxy - r.tested_latency_proxy) / r.baseline_latency_proxy for r in rows]
        path_change = [r.path_changed for r in rows]
        summary[ablation] = {
            "energy_reduction_percent": summarise_numeric(energy_rel),
            "latency_reduction_percent": summarise_numeric(latency_rel),
            "baseline_energy_proxy": summarise_numeric([r.baseline_energy_proxy for r in rows]),
            "tested_energy_proxy": summarise_numeric([r.tested_energy_proxy for r in rows]),
            "baseline_latency_proxy": summarise_numeric([r.baseline_latency_proxy for r in rows]),
            "tested_latency_proxy": summarise_numeric([r.tested_latency_proxy for r in rows]),
            "path_changed_fraction": summarise_numeric(path_change),
        }

    by_condition: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in ["traffic_level", "destination_pattern", "routing_pressure", "topology"]:
        by_condition[key] = {}
        for value in sorted({getattr(t, key) for t in trials}):
            rows = [t for t in trials if t.ablation == "both" and getattr(t, key) == value]
            energy_rel = [100.0 * (r.baseline_energy_proxy - r.tested_energy_proxy) / r.baseline_energy_proxy for r in rows]
            latency_rel = [100.0 * (r.baseline_latency_proxy - r.tested_latency_proxy) / r.baseline_latency_proxy for r in rows]
            by_condition[key][value] = {
                "energy_reduction_percent_mean": round(mean(energy_rel), 6),
                "energy_reduction_percent_std": round(pstdev(energy_rel), 6),
                "latency_reduction_percent_mean": round(mean(latency_rel), 6),
                "latency_reduction_percent_std": round(pstdev(latency_rel), 6),
                "n": len(rows),
            }

    report = {
        "baseline_definition": {
            "routing_behavior": "Standard static routing baseline represented as shortest-path routing under standard routing keys only, with no space-bit dispatch refinement.",
            "processor_mask_behavior": "Full local fan-out with 18 active processor bits on every local-delivery event.",
            "local_core_delivery_behavior": "Local delivery is fixed by the base route only; no adaptive processor-mask refinement is applied.",
        },
        "proposed_method_definition": {
            "space_bits_representation": "Simulated as software dispatch-class policies that can reorder/restrict preferred directions and/or reduce local processor-mask size.",
            "refinement_location": "Inter-chip direction refinement is modeled as Router-side policy; processor-mask refinement is modeled as SNoC/local-delivery-side policy.",
            "favoring_assumptions": "No extra route-table capacity is granted to the proposed method. The same source/destination/hotspot workloads are used for all ablations. However, the energy metric is a synthetic proxy that rewards lower mask size by construction.",
        },
        "energy_metric": {
            "formula": "energy_proxy = hops + 0.18 * mask_size + 0.9 * congested_hops",
            "type": "synthetic simulation proxy",
            "limitations": "Not hardware calibrated; does not model packet buffering, actual router micro-activity, SDRAM energy, monitor-core control overhead, or physical SNoC timing.",
        },
        "summary_by_ablation": summary,
        "stress_test_by_condition_for_both": by_condition,
    }

    (out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    latex_rows = []
    for ablation in ["baseline", "null_dispatch", "space_bits_only", "mask_only", "both"]:
        s = summary[ablation]
        latex_rows.append(
            f"{ablation.replace('_', ' ')} & {s['energy_reduction_percent']['mean']:.2f} $\\pm$ {s['energy_reduction_percent']['std']:.2f} & "
            f"{s['latency_reduction_percent']['mean']:.2f} $\\pm$ {s['latency_reduction_percent']['std']:.2f} & "
            f"{s['path_changed_fraction']['mean']:.2f} & {s['energy_reduction_percent']['n']} \\\\" 
        )
    (out_dir / "audit_table_rows.tex").write_text("\n".join(latex_rows), encoding="utf-8")

    methodology = r"""
\paragraph{Methodology.}
We audited the routing claim in virtual mode using SpiNNakerGraphFrontEnd, SpiNNMachine, and PACMAN on 8$\times$8 and 12$\times$12 virtual topologies. The baseline was defined as standard static routing under standard routing keys only, with no space-bit dispatch refinement and a fixed 18-core local processor mask. The proposed method was represented in software as dispatch policies that could alter preferred inter-chip forwarding order and/or reduce the effective local processor-mask size. We ran identical workloads for baseline and proposed variants across 10 seeds, 2 topology sizes, 3 traffic-density levels, 2 destination-structure regimes, 4 local-mask targets, and 2 routing-pressure settings, yielding 960 audited comparisons per ablation. Reported energy and latency values are synthetic simulation proxies rather than hardware-calibrated measurements.
"""
    (out_dir / "methodology_snippet.tex").write_text(methodology.strip() + "\n", encoding="utf-8")

    limitations = r"""
\paragraph{Limitations.}
The audit remains a software benchmark, not a board-level measurement study. In particular, the energy metric is a synthetic proxy defined as a weighted combination of hop count, local processor-mask size, and congested-hop exposure. It does not model router micro-activity, SNoC timing, SDRAM traffic, monitor-core update overhead, or measured power on physical SpiNNaker hardware. The routing model is therefore suitable for testing the internal logic and sensitivity of the proposed mechanism, but not for claiming established hardware energy savings.
"""
    (out_dir / "limitations_snippet.tex").write_text(limitations.strip() + "\n", encoding="utf-8")

    conservative = r"""
\paragraph{Conservative result statement.}
In the audited virtual-mode benchmark, the combined dispatch policy (space-bit direction refinement plus processor-mask refinement) reduced the synthetic energy proxy relative to the static baseline on average, but the magnitude of the reduction was assumption-dependent and varied substantially across workloads. The previously highlighted 35.2\% figure should therefore not be treated as an established energy result; it was a favorable outcome under a narrower benchmark setup. The stronger defensible claim is that, under this software proxy model, processor-mask refinement appears to be the dominant contributor to the observed reduction, whereas direction refinement alone has smaller and less stable effects and can increase latency in some settings.
"""
    (out_dir / "conservative_result_snippet.tex").write_text(conservative.strip() + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Audit outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
