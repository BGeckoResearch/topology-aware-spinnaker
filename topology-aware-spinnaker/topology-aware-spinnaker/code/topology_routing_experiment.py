from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import spinnaker_graph_front_end as gfe
from spinn_machine.virtual_machine import virtual_machine

Coord = Tuple[int, int]


@dataclass
class RouteReport:
    source: Coord
    destination: Coord
    baseline_path: List[Coord]
    topology_aware_path: List[Coord]
    baseline_hops: int
    topology_aware_hops: int
    candidate_local_cores: List[int]
    selected_local_core: int


def torus_neighbors(x: int, y: int, width: int, height: int) -> List[Coord]:
    return [
        ((x + 1) % width, y),
        ((x - 1) % width, y),
        (x, (y + 1) % height),
        (x, (y - 1) % height),
    ]


def bfs_route(source: Coord, destination: Coord, width: int, height: int, avoid: Coord | None = None) -> List[Coord]:
    q = deque([source])
    parents: Dict[Coord, Coord | None] = {source: None}
    while q:
        node = q.popleft()
        if node == destination:
            break
        for nxt in torus_neighbors(node[0], node[1], width, height):
            if avoid is not None and nxt == avoid:
                continue
            if nxt not in parents:
                parents[nxt] = node
                q.append(nxt)
    if destination not in parents:
        return []
    path = []
    cur = destination
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    return list(reversed(path))


def choose_local_core(packet_class: str, congestion_score: float) -> Tuple[List[int], int]:
    available = list(range(1, 17))
    if packet_class == "latency_critical":
        preferred = [1, 2, 3, 4]
    elif packet_class == "bulk":
        preferred = [9, 10, 11, 12, 13, 14, 15, 16]
    else:
        preferred = [5, 6, 7, 8]

    if congestion_score > 0.7:
        selected = preferred[-1]
    elif congestion_score > 0.3:
        selected = preferred[len(preferred) // 2]
    else:
        selected = preferred[0]
    return available, selected


def main() -> None:
    gfe.setup(n_chips_required=1)
    vm = virtual_machine(8, 8)

    width = vm.width
    height = vm.height
    source = (0, 0)
    destination = (5, 3)
    hot_spot = (1, 0)

    baseline = bfs_route(source, destination, width, height)
    topology_aware = bfs_route(source, destination, width, height, avoid=hot_spot)

    candidate_cores, selected_core = choose_local_core(
        packet_class="latency_critical",
        congestion_score=0.62,
    )

    report = RouteReport(
        source=source,
        destination=destination,
        baseline_path=baseline,
        topology_aware_path=topology_aware,
        baseline_hops=max(0, len(baseline) - 1),
        topology_aware_hops=max(0, len(topology_aware) - 1),
        candidate_local_cores=candidate_cores,
        selected_local_core=selected_core,
    )

    print("=== SpiNNaker Topology-Aware Routing Experiment ===")
    print(f"Virtual machine: {width}x{height}, chips={vm.n_chips}")
    print(f"Source: {report.source}")
    print(f"Destination: {report.destination}")
    print(f"Hot-spot avoided by topology-aware route: {hot_spot}")
    print()
    print(f"Baseline path ({report.baseline_hops} hops): {report.baseline_path}")
    print(f"Topology-aware path ({report.topology_aware_hops} hops): {report.topology_aware_path}")
    print()
    print(f"Candidate local cores: {report.candidate_local_cores}")
    print(f"Selected local core: {report.selected_local_core}")
    print()
    print("Interpretation:")
    print("- baseline route gives a shortest-path reference")
    print("- topology-aware route demonstrates simple congestion avoidance")
    print("- local-core selection shows where packet classification / adaptive policies plug in")

    gfe.stop()


if __name__ == "__main__":
    main()
