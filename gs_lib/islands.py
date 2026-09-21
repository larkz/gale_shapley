"""
islands.py
Structural machinery for PrefLID: partition structures, configuration
enumeration, targeted round-robin refinement, lattice construction,
and island decomposition.

Design references:
    - Partition structure derived from pairwise confidence intervals.
    - Configuration count |Omega| = product of |block|! over all agents.
    - Lattice from a fully-resolved configuration via GS + rotations.
    - Island = group of lattices sharing a common stable matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from itertools import permutations, product as cartesian_product

from gs_lib.gs_tools import (
    Man, Woman, Matching, PreferenceList, GaleShapley, StableMatchingLattice,
)
from gs_lib.bt import _canonical


# ============================================================================
# Partition structure
# ============================================================================

@dataclass
class PartitionStructure:
    """Per-agent partition structure derived from current CIs.

    partitions[agent] = list of blocks. Each block is a list of partners
    whose confidence intervals pairwise overlap (so their internal order
    is unresolved). Between two consecutive blocks there is a "cut":
    every partner in the earlier block is confidently preferred over every
    partner in the later block.
    """
    partitions: Dict[Hashable, List[List[Hashable]]]

    def n_cuts(self, agent: Hashable) -> int:
        return len(self.partitions[agent]) - 1

    def is_fully_resolved(self) -> bool:
        return all(
            len(block) == 1
            for blocks in self.partitions.values()
            for block in blocks
        )

    def size_of_omega(self) -> int:
        """Product of |block|! over all blocks and all agents."""
        from math import factorial
        total = 1
        for blocks in self.partitions.values():
            for block in blocks:
                total *= factorial(len(block))
                # Bail early if astronomically large — useful for
                # comparisons against a budget.
                if total > 10**18:
                    return total
        return total


def build_partitions_from_cis(
    partners: Sequence[Hashable],
    theta_hat: Dict[Hashable, float],
    ci: Dict[Tuple[Hashable, Hashable], Tuple[float, float]],
    counts: Optional[Dict[Tuple[Hashable, Hashable], int]] = None,
    min_samples_per_cut: int = 10,
) -> List[List[Hashable]]:
    """Partition a single agent's partners into blocks using CIs.

    Method:
        1. Sort partners by theta_hat descending.
        2. Walk down the sorted list; at each adjacent pair (b_i, b_{i+1}),
           look up their CI on the pairwise preference p_hat_{b_i, b_{i+1}}.
           If the interval excludes 0.5 AND the pair has at least
           `min_samples_per_cut` samples, they are "cut" apart.
           Otherwise, they stay in the same block.

    The CI dict is keyed by _canonical(b_i, b_j), matching PairCounts.
    If `counts` is provided, pairs with fewer than `min_samples_per_cut`
    samples are never used to place a cut, guarding against spurious cuts
    from small-sample CI noise.
    """
    ranked = sorted(partners, key=lambda b: (-theta_hat.get(b, 0.0), str(b)))
    if not ranked:
        return []

    blocks: List[List[Hashable]] = [[ranked[0]]]
    for i in range(len(ranked) - 1):
        b_i, b_j = ranked[i], ranked[i + 1]
        key = _canonical(b_i, b_j)

        # Refuse to cut unless this pair has enough samples.
        if counts is not None:
            n = counts.get(key, 0)
            if n < min_samples_per_cut:
                blocks[-1].append(b_j)
                continue

        lo, hi = ci.get(key, (0.0, 1.0))    # default: unresolved
        cut = (lo > 0.5) or (hi < 0.5)
        if cut:
            blocks.append([b_j])
        else:
            blocks[-1].append(b_j)
    return blocks


# ============================================================================
# Configuration enumeration
# ============================================================================

def enumerate_configurations(
    structure: PartitionStructure,
    budget: int,
) -> Optional[List[Dict[Hashable, List[Hashable]]]]:
    """Enumerate all total orders compatible with the partition structure.

    Returns:
        - A list of configurations (each a dict agent -> ordered list of
          partners) if `size_of_omega <= budget`.
        - None if the configuration space exceeds the budget.

    Each configuration is a *fully-resolved* two-sided ordering ready for GS.
    """
    total = structure.size_of_omega()
    if total > budget:
        return None

    per_agent_orders: Dict[Hashable, List[List[Hashable]]] = {}
    for agent, blocks in structure.partitions.items():
        block_perms = [list(permutations(block)) for block in blocks]
        agent_orders: List[List[Hashable]] = []
        for combo in cartesian_product(*block_perms):
            ordered: List[Hashable] = []
            for block_perm in combo:
                ordered.extend(block_perm)
            agent_orders.append(ordered)
        per_agent_orders[agent] = agent_orders

    agents = list(per_agent_orders.keys())
    configs: List[Dict[Hashable, List[Hashable]]] = []
    for combo in cartesian_product(*(per_agent_orders[a] for a in agents)):
        config = {a: list(combo[i]) for i, a in enumerate(agents)}
        configs.append(config)
    return configs


# ============================================================================
# Lattice from configuration
# ============================================================================

def lattice_from_configuration(
    config: Dict[Hashable, List[Hashable]],
    max_vertices: int = 5000,
) -> Optional[frozenset]:
    """Build the lattice of stable matchings for a fully-resolved config.

    Returns a frozenset of Matching objects, or None if the lattice would
    exceed `max_vertices`.

    Uses the existing StableMatchingLattice from gs_tools.
    """
    prefs_dict: Dict[Hashable, List[Hashable]] = {}
    for agent, ordered in config.items():
        prefs_dict[agent] = list(ordered)

    prefs = PreferenceList(prefs_dict)
    lattice = StableMatchingLattice(prefs)
    man_opt, woman_opt = lattice.build_lattice()

    all_matchings = lattice.get_all_matchings()
    if len(all_matchings) > max_vertices:
        return None

    return frozenset(all_matchings)


# ============================================================================
# Island construction
# ============================================================================

@dataclass
class Island:
    """A set of lattices sharing a common stable matching."""
    lattice_indices: List[int]                 # indices into the input list
    H_star: Matching                           # centroidal matching
    support: int                               # number of lattices sharing H_star
    support_max: int = 0                       # max support across vertices
    support_min: int = 0                       # min support across vertices
    support_mean: float = 0.0                  # mean support across vertices


def construct_islands(
    lattices: List[frozenset],
) -> List[Island]:
    """Group lattices into islands by shared vertices.

    Two lattices are in the same island iff their vertex sets intersect.
    Islands are computed via union-find over lattices. Within each island,
    the centroidal matching H_star is the vertex covered by the most
    lattices (ties broken by str).

    Parameters:
        lattices: list of frozensets of Matching.

    Returns:
        list of Island objects.
    """
    n = len(lattices)
    if n == 0:
        return []

    # Union-find over lattice indices.
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Inverted index: vertex -> list of lattice indices containing it.
    vertex_to_lattices: Dict[Matching, List[int]] = {}
    for i, V in enumerate(lattices):
        for v in V:
            vertex_to_lattices.setdefault(v, []).append(i)

    # Union lattices that share any vertex.
    for vertex, idxs in vertex_to_lattices.items():
        for k in range(1, len(idxs)):
            union(idxs[0], idxs[k])

    # Group by root.
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    islands: List[Island] = []
    for root, idxs in groups.items():
        # Count how many lattices in this group contain each vertex.
        support: Dict[Matching, int] = {}
        for i in idxs:
            for v in lattices[i]:
                support[v] = support.get(v, 0) + 1
        # H_star = max-support vertex, tie-break by str.
        H_star = max(
            support.keys(),
            key=lambda v: (support[v], str(v)),
        )
        support_values = list(support.values())
        islands.append(Island(
            lattice_indices=sorted(idxs),
            H_star=H_star,
            support=support[H_star],
            support_max=max(support_values),
            support_min=min(support_values),
            support_mean=sum(support_values) / len(support_values),
        ))

    return islands