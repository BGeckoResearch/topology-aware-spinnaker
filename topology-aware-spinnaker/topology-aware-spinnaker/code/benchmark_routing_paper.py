from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import spinnaker_graph_front_end as gfe
from spinn_machine.virtual_machine import virtual_machine

Coord = Tuple[int, int]
Direction = str

DIRECTIONS: Sequence[Direction] = ("E", "W", "N", "S")
DIR_TO_STEP = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, 1),
    "S": (0, -1),
}

# Dispatch classes used for the benchmark.
DISPATCH_MAP = {
    "null": {
        "directions": {"E", "W", "N", "S"},
        "mask_size": 18,
        "packet_class": "default",
    },
    "north_east_bias": {
        "directions": {"E", "N", "W"},
        "mask_size": 8,
        "packet_class": "latency_critical",
    },
    "south_west_bias": {
        "directions": {"W", "S", "N"},
        "mask_size": 8,
        "packet_class": "latency_critical",
    },
    "energy_save": {
        "directions": {"E", "W", "N", "S"},
        "mask_size": 4,
        "packet_class": "bulk",
    },
}

SCENARIOS = [
    {"name": "eastbound_hotspot", "source": (0, 0), "destination": (5, 3), "hotspots": {(7, 0), (6, 0)}},
    {"name": "northbound_congestion", "source": (2, 1), "destination": (2, 6), "hotspots": {(2, 2), (2, 3), (2, 4)}},
    {"name": "diagonal_pressure", "source": (1, 1), "destination": (6, 5), "hotspots": {(2, 1), (3, 2), (4, 3)}},
    {"name": "return_flow", "source": (6, 5), "destination": (1, 1), "hotspots": {(5, 5), (4, 5), (3, 4)}},
    {"name": "wide_route", "source": (0, 3), "destination": (7, 3), "hotspots": {(7, 3)}},
    {"name": "vertical_wrap", "source": (4, 0), "destination": (4, 7), "hotspots": {(4, 7)}},
]


@dataclass
class TrialResult:
    scenario: str
    dispatch_class: str
    source: Coord
    destination: Coord
    baseline_hops: int
    adaptive_hops: int
    baseline_congested_hops: int
    adaptive_congested_hops: int
    baseline_mask_size: int
    adaptive_mask_size: int
    baseline_energy_score: float
    adaptive_energy_score: float
    baseline_latency_score: float
    adaptive_latency_score: float
    path_changed: bool



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



def bfs_path(source: Coord, destination: Coord, width: int, height: int, allowed_dirs: Iterable[Direction], blocked: set[Coord]) -> List[Coord]:
    allowed_dirs = tuple(allowed_dirs)
    q = deque([source])
    parents: Dict[Coord, Coord | None] = {source: None}
    while q:
        cur = q.popleft()
        if cur == destination:
            break
        for d in allowed_dirs:
            nxt = torus_step(cur, d, width, height)
            if nxt in blocked and nxt != destination:
                continue
            if nxt not in parents:
                parents[nxt] = cur
                q.append(nxt)
    if destination not in parents:
        return []
    path: List[Coord] = []
    cur: Coord | None = destination
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return list(reversed(path))



def count_congested(path: Sequence[Coord], hotspots: set[Coord]) -> int:
    return sum(1 for p in path[1:] if p in hotspots)



def energy_score(hops: int, mask_size: int, congested_hops: int) -> float:
    return round(hops * 1.0 + mask_size * 0.18 + congested_hops * 0.9, 4)



def latency_score(hops: int, congested_hops: int) -> float:
    return round(hops * 1.0 + congested_hops * 1.75, 4)



def adaptive_route(source: Coord, destination: Coord, width: int, height: int, hotspots: set[Coord], dispatch_name: str) -> Tuple[List[Coord], int]:
    dispatch = DISPATCH_MAP[dispatch_name]
    preferred = preferred_directions(source, destination, width, height)
    ordered = [d for d in preferred if d in dispatch["directions"]] + [d for d in preferred if d not in dispatch["directions"]]

    # First attempt: honor dispatch directions only, avoid hotspots.
    restricted = [d for d in ordered if d in dispatch["directions"]]
    path = bfs_path(source, destination, width, height, restricted or ordered, hotspots)
    if not path:
        # Fall back to all directions, still avoid hotspots.
        path = bfs_path(source, destination, width, height, ordered, hotspots)
    if not path:
        # Last resort: ignore hotspots.
        path = bfs_path(source, destination, width, height, ordered, set())
    return path, dispatch["mask_size"]



def run_benchmarks() -> List[TrialResult]:
    gfe.setup(n_chips_required=1)
    vm = virtual_machine(8, 8)
    width, height = vm.width, vm.height
    results: List[TrialResult] = []

    for scenario in SCENARIOS:
        source = scenario["source"]
        destination = scenario["destination"]
        hotspots = set(scenario["hotspots"])

        baseline_path = bfs_path(source, destination, width, height, preferred_directions(source, destination, width, height), set())
        baseline_hops = max(0, len(baseline_path) - 1)
        baseline_congested = count_congested(baseline_path, hotspots)
        baseline_mask = 18
        baseline_energy = energy_score(baseline_hops, baseline_mask, baseline_congested)
        baseline_latency = latency_score(baseline_hops, baseline_congested)

        for dispatch_name in DISPATCH_MAP:
            adaptive_path, adaptive_mask = adaptive_route(source, destination, width, height, hotspots, dispatch_name)
            adaptive_hops = max(0, len(adaptive_path) - 1)
            adaptive_congested = count_congested(adaptive_path, hotspots)
            adaptive_energy = energy_score(adaptive_hops, adaptive_mask, adaptive_congested)
            adaptive_latency = latency_score(adaptive_hops, adaptive_congested)
            results.append(
                TrialResult(
                    scenario=scenario["name"],
                    dispatch_class=dispatch_name,
                    source=source,
                    destination=destination,
                    baseline_hops=baseline_hops,
                    adaptive_hops=adaptive_hops,
                    baseline_congested_hops=baseline_congested,
                    adaptive_congested_hops=adaptive_congested,
                    baseline_mask_size=baseline_mask,
                    adaptive_mask_size=adaptive_mask,
                    baseline_energy_score=baseline_energy,
                    adaptive_energy_score=adaptive_energy,
                    baseline_latency_score=baseline_latency,
                    adaptive_latency_score=adaptive_latency,
                    path_changed=(adaptive_path != baseline_path),
                )
            )
    gfe.stop()
    return results



def summarise(results: Sequence[TrialResult]) -> dict:
    grouped: Dict[str, List[TrialResult]] = {}
    for r in results:
        grouped.setdefault(r.dispatch_class, []).append(r)

    summary = {}
    for dispatch, rows in grouped.items():
        summary[dispatch] = {
            "avg_baseline_energy": round(mean(r.baseline_energy_score for r in rows), 4),
            "avg_adaptive_energy": round(mean(r.adaptive_energy_score for r in rows), 4),
            "avg_baseline_latency": round(mean(r.baseline_latency_score for r in rows), 4),
            "avg_adaptive_latency": round(mean(r.adaptive_latency_score for r in rows), 4),
            "avg_baseline_hops": round(mean(r.baseline_hops for r in rows), 4),
            "avg_adaptive_hops": round(mean(r.adaptive_hops for r in rows), 4),
            "avg_baseline_congested_hops": round(mean(r.baseline_congested_hops for r in rows), 4),
            "avg_adaptive_congested_hops": round(mean(r.adaptive_congested_hops for r in rows), 4),
            "fraction_paths_changed": round(mean(1.0 if r.path_changed else 0.0 for r in rows), 4),
        }
    return summary



def write_outputs(results: Sequence[TrialResult], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "routing_benchmark_results.csv"
    json_path = out_dir / "routing_benchmark_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")



def main() -> None:
    out_dir = Path(__file__).with_name("benchmark_outputs")
    results = run_benchmarks()
    summary = summarise(results)
    write_outputs(results, summary, out_dir)
    print(json.dumps(summary, indent=2))
    print(f"Detailed results written to: {out_dir}")


if __name__ == "__main__":
    main()
