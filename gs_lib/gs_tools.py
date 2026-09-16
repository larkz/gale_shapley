"""
Stable Matching with Rotations - Complete Implementation
Handles unequal numbers of participants on each side.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, FrozenSet, Generic, TypeVar
from dataclasses import dataclass
from collections import defaultdict, deque
import itertools

# ============================================================================
# MODULE 1: Abstract Base Classes and Core Data Structures
# ============================================================================

T = TypeVar('T')

@dataclass(frozen=True)
class Participant(ABC):
    """Abstract base class for participants in the matching."""
    id: str
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Participant):
            return False
        return self.id == other.id
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

class Man(Participant):
    """Represents a man in the matching problem."""
    pass

class Woman(Participant):
    """Represents a woman in the matching problem."""
    pass

@dataclass(frozen=True)
class MatchPair:
    """Represents a matched pair."""
    man: Man
    woman: Woman
    
    def __repr__(self):
        return f"({self.man.id}-{self.woman.id})"

@dataclass(frozen=True)
class Matching:
    """
    Represents a complete matching (potentially with unmatched participants).
    Immutable and hashable for use in lattice structures.
    """
    pairs: FrozenSet[MatchPair]
    unmatched_men: FrozenSet[Man]
    unmatched_women: FrozenSet[Woman]
    
    @classmethod
    def from_dict(cls, matching_dict: Dict[Man, Optional[Woman]], 
                  all_men: Set[Man], all_women: Set[Woman]):
        """Create a Matching from a dictionary of men to optional women."""
        pairs = set()
        matched_men = set()
        matched_women = set()
        
        for man, woman in matching_dict.items():
            if woman is not None:
                pairs.add(MatchPair(man, woman))
                matched_men.add(man)
                matched_women.add(woman)
        
        unmatched_men = frozenset(all_men - matched_men)
        unmatched_women = frozenset(all_women - matched_women)
        
        return cls(frozenset(pairs), unmatched_men, unmatched_women)
    
    def get_partner(self, participant: Participant) -> Optional[Participant]:
        """Get the partner of a participant, or None if unmatched."""
        if isinstance(participant, Man):
            for pair in self.pairs:
                if pair.man == participant:
                    return pair.woman
            return None
        else:  # Woman
            for pair in self.pairs:
                if pair.woman == participant:
                    return pair.man
            return None
    
    def to_dict(self) -> Dict[Man, Optional[Woman]]:
        """Convert to dictionary representation."""
        result = {}
        all_men = set(pair.man for pair in self.pairs) | set(self.unmatched_men)
        for man in all_men:
            partner = self.get_partner(man)
            result[man] = partner
        return result
    
    def __repr__(self):
        pairs_str = ", ".join(str(p) for p in sorted(self.pairs, key=lambda x: x.man.id))
        unmatched_m = ", ".join(m.id for m in sorted(self.unmatched_men, key=lambda x: x.id))
        unmatched_w = ", ".join(w.id for w in sorted(self.unmatched_women, key=lambda x: x.id))
        return f"Matching(pairs=[{pairs_str}], unmatched_men=[{unmatched_m}], unmatched_women=[{unmatched_w}])"

class PreferenceList:
    """
    Represents preference lists for both sides.
    Handles incomplete lists (not all participants need to be ranked).
    """
    def __init__(self, preferences: Dict[Participant, List[Participant]]):
        """
        Args:
            preferences: Dict mapping each participant to their ordered preference list.
                        Lists may be incomplete (some participants may not be ranked).
        """
        self.preferences = preferences
        self.rankings = {}
        
        # Build ranking maps for O(1) lookup
        for person, pref_list in preferences.items():
            self.rankings[person] = {p: i for i, p in enumerate(pref_list)}
    
    def get_preference(self, person: Participant) -> List[Participant]:
        """Get the preference list for a participant."""
        return self.preferences.get(person, [])
    
    def get_rank(self, person: Participant, candidate: Participant) -> Optional[int]:
        """
        Get the rank of candidate in person's preference list.
        Returns None if candidate is not ranked.
        Lower rank = more preferred.
        """
        return self.rankings.get(person, {}).get(candidate)
    
    def prefers(self, person: Participant, candidate1: Participant, candidate2: Participant) -> bool:
        """
        Check if person prefers candidate1 over candidate2.
        Returns True if candidate1 is ranked and candidate2 is not ranked.
        """
        rank1 = self.get_rank(person, candidate1)
        rank2 = self.get_rank(person, candidate2)
        
        if rank1 is None and rank2 is None:
            return False
        if rank1 is not None and rank2 is None:
            return True
        if rank1 is None and rank2 is not None:
            return False
        
        return rank1 < rank2
    
    def is_acceptable(self, person: Participant, candidate: Participant) -> bool:
        """Check if candidate is in person's preference list."""
        return candidate in self.rankings.get(person, {})


# ============================================================================
# MODULE 2: Gale-Shapley Algorithm
# ============================================================================

class GaleShapley:
    """
    Implementation of the Gale-Shapley algorithm for finding stable matchings.
    Handles unequal numbers and incomplete preference lists.
    """
    
    def __init__(self, preferences: PreferenceList):
        self.preferences = preferences
    
    def find_stable_matching(self, proposing_side: str = "men") -> Matching:
        """
        Find a stable matching using the Gale-Shapley algorithm.
        
        Args:
            proposing_side: "men" or "women" - which side proposes
        
        Returns:
            A stable Matching optimal for the proposing side.
        """
        if proposing_side == "men":
            return self._men_propose()
        else:
            return self._women_propose()
    
    def _men_propose(self) -> Matching:
        """Men-proposing Gale-Shapley algorithm."""
        # Extract all men and women from preference lists
        all_men = {p for p in self.preferences.preferences if isinstance(p, Man)}
        all_women = {p for p in self.preferences.preferences if isinstance(p, Woman)}
        
        # Initialize: all men free, all women free
        free_men = deque(all_men)
        current_matches: Dict[Woman, Optional[Man]] = {w: None for w in all_women}
        proposals: Dict[Man, int] = {m: 0 for m in all_men}  # Next proposal index
        
        while free_men:
            man = free_men[0]
            pref_list = self.preferences.get_preference(man)
            
            # If man has exhausted his preference list, he remains unmatched
            if proposals[man] >= len(pref_list):
                free_men.popleft()  # Remove from free list, will be unmatched
                continue
            
            # Get next woman to propose to
            woman = pref_list[proposals[man]]
            proposals[man] += 1
            
            if current_matches[woman] is None:
                # Woman is free - accept proposal
                current_matches[woman] = man
                free_men.popleft()
            else:
                # Woman is engaged - check if she prefers new proposal
                current_man = current_matches[woman]
                if self.preferences.prefers(woman, man, current_man):
                    # She prefers the new man
                    current_matches[woman] = man
                    free_men.popleft()
                    free_men.append(current_man)  # Old partner becomes free
                # If she prefers current partner, man stays free and tries next woman
        
        # Build the matching
        matching_dict = {}
        matched_women = set()
        
        for woman, man in current_matches.items():
            if man is not None:
                matching_dict[man] = woman
                matched_women.add(woman)
        
        # Add unmatched men
        for man in all_men:
            if man not in matching_dict:
                matching_dict[man] = None
        
        return Matching.from_dict(matching_dict, all_men, all_women)
    
    def _women_propose(self) -> Matching:
        """Women-proposing Gale-Shapley algorithm."""
        all_men = {p for p in self.preferences.preferences if isinstance(p, Man)}
        all_women = {p for p in self.preferences.preferences if isinstance(p, Woman)}
        
        free_women = deque(all_women)
        current_matches: Dict[Man, Optional[Woman]] = {m: None for m in all_men}
        proposals: Dict[Woman, int] = {w: 0 for w in all_women}
        
        while free_women:
            woman = free_women[0]
            pref_list = self.preferences.get_preference(woman)
            
            if proposals[woman] >= len(pref_list):
                free_women.popleft()
                continue
            
            man = pref_list[proposals[woman]]
            proposals[woman] += 1
            
            if current_matches[man] is None:
                current_matches[man] = woman
                free_women.popleft()
            else:
                current_woman = current_matches[man]
                if self.preferences.prefers(man, woman, current_woman):
                    current_matches[man] = woman
                    free_women.popleft()
                    free_women.append(current_woman)
        
        matching_dict = {}
        matched_men = set()
        
        for man, woman in current_matches.items():
            if woman is not None:
                matching_dict[man] = woman
                matched_men.add(man)
        
        all_men_set = all_men
        all_women_set = all_women
        
        for man in all_men_set:
            if man not in matching_dict:
                matching_dict[man] = None
        
        return Matching.from_dict(matching_dict, all_men_set, all_women_set)
    
    def find_both_optimal(self) -> Tuple[Matching, Matching]:
        """Find both man-optimal and woman-optimal stable matchings."""
        man_optimal = self.find_stable_matching("men")
        woman_optimal = self.find_stable_matching("women")
        return man_optimal, woman_optimal


# ============================================================================
# MODULE 3: Stability Verifier
# ============================================================================

class StabilityVerifier:
    """
    Verifies whether a given matching is stable.
    """
    
    def __init__(self, preferences: PreferenceList):
        self.preferences = preferences
    
    def is_stable(self, matching: Matching) -> Tuple[bool, Optional[str], Optional[Tuple]]:
        """
        Check if a matching is stable.
        
        Returns:
            (is_stable, reason, blocking_pair)
            If stable: (True, None, None)
            If unstable: (False, reason, (man, woman) blocking pair)
        """
        # Check 1: Individual rationality - no one is matched to someone not on their list
        for pair in matching.pairs:
            if not self.preferences.is_acceptable(pair.man, pair.woman):
                return False, f"{pair.man.id} is matched to unacceptable partner {pair.woman.id}", (pair.man, pair.woman)
            if not self.preferences.is_acceptable(pair.woman, pair.man):
                return False, f"{pair.woman.id} is matched to unacceptable partner {pair.man.id}", (pair.woman, pair.man)
        
        # Check 2: No blocking pairs
        # A blocking pair (m, w) exists if:
        # - m and w are not matched to each other
        # - m prefers w to his current partner (or is unmatched and finds w acceptable)
        # - w prefers m to her current partner (or is unmatched and finds m acceptable)
        
        all_men = set(pair.man for pair in matching.pairs) | set(matching.unmatched_men)
        all_women = set(pair.woman for pair in matching.pairs) | set(matching.unmatched_women)
        
        for man in all_men:
            man_partner = matching.get_partner(man)
            
            # Check women in man's preference list up to his current partner
            for woman in self.preferences.get_preference(man):
                if woman == man_partner:
                    break  # Only check women preferred to current partner
                
                woman_partner = matching.get_partner(woman)
                
                # Check if woman prefers this man
                woman_prefs_man = self.preferences.prefers(woman, man, woman_partner) if woman_partner else self.preferences.is_acceptable(woman, man)
                
                if woman_prefs_man:
                    return False, f"Blocking pair: ({man.id}, {woman.id})", (man, woman)
        
        # Also check from women's perspective (for unmatched men looking at unmatched women)
        for woman in all_women:
            woman_partner = matching.get_partner(woman)
            
            for man in self.preferences.get_preference(woman):
                if man == woman_partner:
                    break
                
                man_partner = matching.get_partner(man)
                
                man_prefs_woman = self.preferences.prefers(man, woman, man_partner) if man_partner else self.preferences.is_acceptable(man, woman)
                
                if man_prefs_woman:
                    return False, f"Blocking pair: ({man.id}, {woman.id})", (man, woman)
        
        return True, None, None


# ============================================================================
# MODULE 4: Single Rotation Operation
# ============================================================================

class RotationOperator:
    """
    Performs rotations to move between stable matchings.
    Handles unequal numbers by focusing only on matched participants.
    """
    
    def __init__(self, preferences: PreferenceList):
        self.preferences = preferences
    
    def find_exposed_rotations(self, matching: Matching) -> List[List[Tuple[Man, Woman]]]:
        """
        Find all rotations exposed in the current stable matching.

        A rotation is a cycle (m1,w1), (m2,w2), ..., (mk,wk) where:
        - Each (mi, wi) is a pair in the matching
        - For each i, w_{i+1} is the first woman on mi's list after wi
        who prefers mi to her current partner

        Unmatched women are always willing (they have no current partner),
        so they can serve as next-best partners and terminate a chain
        rather than participating in a cycle.

        Returns:
            List of rotations, each rotation is a list of (man, woman) pairs
        """
        # Build dictionary mapping each person to their partner.
        # Unmatched participants are NOT keys here.
        partners: Dict[Hashable, Hashable] = {}
        for pair in matching.pairs:
            partners[pair.man] = pair.woman
            partners[pair.woman] = pair.man

        # For each matched man, find his next-best willing woman.
        # An unmatched woman is always willing; a matched woman is willing
        # iff she prefers the man over her current partner.
        next_best: Dict[Man, Optional[Woman]] = {}

        for pair in matching.pairs:
            man = pair.man
            current_woman = pair.woman

            pref_list = self.preferences.get_preference(man)
            try:
                start_idx = pref_list.index(current_woman) + 1
            except ValueError:
                # Should not happen for a stable matching.
                next_best[man] = None
                continue

            found = False
            for woman in pref_list[start_idx:]:
                current_man_of_woman = partners.get(woman)   # None if unmatched
                if current_man_of_woman is None:
                    # Unmatched woman — always willing.
                    next_best[man] = woman
                    found = True
                    break
                if self.preferences.prefers(woman, man, current_man_of_woman):
                    next_best[man] = woman
                    found = True
                    break

            if not found:
                next_best[man] = None

        # Find cycles in the directed graph on men, where an edge
        # m -> partners[next_best[m]] exists if next_best[m] is a matched woman.
        # If next_best[m] is unmatched, the chain dead-ends — no cycle through m.
        rotations: List[List[Tuple[Man, Woman]]] = []
        globally_visited: set = set()

        for start_man in next_best:
            if start_man in globally_visited or next_best[start_man] is None:
                continue

            # Walk this chain until we hit a visited man, a dead-end, or
            # a repeat of a man already on this path.
            path: List[Man] = []
            path_index: Dict[Man, int] = {}
            current_man: Optional[Man] = start_man

            while (current_man is not None
                and current_man not in globally_visited
                and current_man in next_best
                and next_best[current_man] is not None):

                if current_man in path_index:
                    # Cycle found: extract from the first occurrence.
                    cycle_start = path_index[current_man]
                    cycle_men = path[cycle_start:]
                    rotation = [(m, partners[m]) for m in cycle_men]
                    if len(rotation) >= 2:
                        rotations.append(rotation)
                    globally_visited.update(cycle_men)
                    break

                path_index[current_man] = len(path)
                path.append(current_man)

                next_woman = next_best[current_man]
                # If the next-best woman is unmatched, the chain ends here.
                if next_woman not in partners:
                    break

                current_man = partners[next_woman]

            # Mark everything on this path as visited so we don't re-walk it.
            globally_visited.update(path)

        return rotations
    
    def perform_rotation(self, matching: Matching, rotation: List[Tuple[Man, Woman]]) -> Matching:
        """
        Perform a rotation to get a new stable matching.
        
        Args:
            matching: Current stable matching
            rotation: List of (man, woman) pairs forming the rotation cycle
        
        Returns:
            New stable matching after performing the rotation
        """
        # Build dictionary of current matches
        current_dict = {}
        for pair in matching.pairs:
            current_dict[pair.man] = pair.woman
        
        # Perform the rotation: each man marries the next woman in the cycle
        new_dict = current_dict.copy()
        
        for i, (man, _) in enumerate(rotation):
            # Each man marries the woman from the next pair in the cycle
            next_woman = rotation[(i + 1) % len(rotation)][1]
            new_dict[man] = next_woman
        
        # Get all men and women
        all_men = set(pair.man for pair in matching.pairs) | set(matching.unmatched_men)
        all_women = set(pair.woman for pair in matching.pairs) | set(matching.unmatched_women)
        
        return Matching.from_dict(new_dict, all_men, all_women)

    # def perform_rotation(self, matching: Matching, rotation: List[Tuple[Man, Woman]]) -> Matching:
    #     """
    #     Eliminating rotation R = ((m_0, w_0), (m_1, w_1), ..., (m_{k-1}, w_{k-1}))
    #     reassigns m_i to w_{i+1 (mod k)}.
    #     """
    #     new_dict = {pair.man: pair.woman for pair in matching.pairs}
        
    #     n = len(rotation)
    #     for i in range(n):
    #         man = rotation[i][0]
    #         # m_i receives the woman from the NEXT pair in the cycle (w_{i+1})
    #         next_woman = rotation[(i + 1) % n][1]
    #         new_dict[man] = next_woman

    #     all_men = set(pair.man for pair in matching.pairs) | set(matching.unmatched_men)
    #     all_women = set(pair.woman for pair in matching.pairs) | set(matching.unmatched_women)
        
    #     return Matching.from_dict(new_dict, all_men, all_women)
    
    def get_rotation_effect(self, rotation: List[Tuple[Man, Woman]]) -> str:
        """Describe the effect of a rotation on participants."""
        description = []
        for i, (man, old_woman) in enumerate(rotation):
            new_woman = rotation[(i + 1) % len(rotation)][1]
            old_rank = self.preferences.get_rank(man, old_woman)
            new_rank = self.preferences.get_rank(man, new_woman)
            
            description.append(
                f"  {man.id}: {old_woman.id}(#{old_rank+1}) → {new_woman.id}(#{new_rank+1}) "
                f"[goes {'DOWN' if new_rank > old_rank else 'UP'}]"
            )
        
        return "\n".join(description)


# ============================================================================
# MODULE 5: Stable Matching Lattice
# ============================================================================

@dataclass
class LatticeNode:
    """A node in the stable matching lattice."""
    matching: Matching
    rotations: List[List[Tuple[Man, Woman]]]  # Rotations exposed at this node
    parents: List['LatticeNode']  # Nodes that can reach this via rotation
    children: List['LatticeNode']  # Nodes reachable from this via rotation
    
    def __hash__(self):
        return hash(self.matching)

class StableMatchingLattice:
    """
    Builds and represents the lattice of all stable matchings.
    Uses rotations to explore the complete lattice structure.
    """
    
    def __init__(self, preferences: PreferenceList):
        self.preferences = preferences
        self.gs = GaleShapley(preferences)
        self.rotator = RotationOperator(preferences)
        self.verifier = StabilityVerifier(preferences)
        
        self.nodes: Dict[Matching, LatticeNode] = {}
        self.man_optimal: Optional[LatticeNode] = None
        self.woman_optimal: Optional[LatticeNode] = None
    
    def build_lattice(self) -> Tuple[LatticeNode, LatticeNode]:
        """
        Build the complete lattice of stable matchings.
        
        Returns:
            (man_optimal_node, woman_optimal_node)
        """
        # Find the two extremes
        man_opt_matching, woman_opt_matching = self.gs.find_both_optimal()
        
        # Create nodes for both
        self.man_optimal = self._get_or_create_node(man_opt_matching)
        self.woman_optimal = self._get_or_create_node(woman_opt_matching)
        
        # Use BFS from man-optimal to discover all stable matchings
        queue = deque([self.man_optimal])
        visited = {self.man_optimal.matching}
        
        while queue:
            current_node = queue.popleft()
            current_matching = current_node.matching
            
            # Find all rotations exposed at this node
            rotations = self.rotator.find_exposed_rotations(current_matching)
            current_node.rotations = rotations
            
            # Apply each rotation to get child nodes
            for rotation in rotations:
                new_matching = self.rotator.perform_rotation(current_matching, rotation)
                
                # Verify the new matching is stable
                is_stable, _, _ = self.verifier.is_stable(new_matching)
                if not is_stable:
                    print(f"Warning: Rotation produced unstable matching!")
                    print(f"Current matching: {current_matching}")
                    print(f"Rotation: {rotation}")
                    print(f"New matching: {new_matching}")

                    print("=== Rotation instability diagnostic ===")
                    print("Verifier prefs id:", id(self.verifier.preferences))
                    print("Rotator  prefs id:", id(self.rotator.preferences))
                    print("Same object?     ", self.verifier.preferences is self.rotator.preferences)

                    # Rotator's preferences for everyone in the rotation
                    print("--- Rotator's preferences ---")
                    for m, w in rotation:
                        print(f"  prefs({m.id}) = "
                            f"{[x.id for x in self.rotator.preferences.get_preference(m)]}")
                        print(f"  prefs({w.id}) = "
                            f"{[x.id for x in self.rotator.preferences.get_preference(w)]}")

                    # Verifier's preferences for everyone in the rotation
                    print("--- Verifier's preferences ---")
                    for m, w in rotation:
                        print(f"  prefs({m.id}) = "
                            f"{[x.id for x in self.verifier.preferences.get_preference(m)]}")
                        print(f"  prefs({w.id}) = "
                            f"{[x.id for x in self.verifier.preferences.get_preference(w)]}")

                    # Matching before / after
                    current_dict = {p.man: p.woman for p in current_matching.pairs}
                    new_dict = current_dict.copy()
                    for i, (man, _) in enumerate(rotation):
                        next_woman = rotation[(i + 1) % len(rotation)][1]
                        new_dict[man] = next_woman
                    print("Rotation:", [(m.id, w.id) for m, w in rotation])
                    print("Before:  ", {m.id: w.id for m, w in current_dict.items()})
                    print("After:   ", {m.id: w.id for m, w in new_dict.items()})
                    print("Reason:  ", self.reason)

                    # Is the CURRENT matching already stable?
                    ok_current, reason_current, _ = self.verifier.is_stable(current_matching)
                    print("Current matching stable?", ok_current, reason_current)
                    print("=======================================")

                    # print(f"  prefs({m.id}) = " f"{[x.id for x in self.preferences.get_preference(m)]}")
                    # print(f"  prefs({w.id}) = " f"{[x.id for x in self.preferences.get_preference(w)]}")

                    # print("Verifier's preference list for m1:",
                    #     [x.id for x in self.verifier.preferences.get_preference(Man("m1"))])
                    # print("Verifier's preference list for m2:",
                    #     [x.id for x in self.verifier.preferences.get_preference(Man("m2"))])
                    # print("Verifier's preference list for w1:",
                    #     [x.id for x in self.verifier.preferences.get_preference(Woman("w1"))])
                    # print("Verifier's preference list for w2:",
                    #     [x.id for x in self.verifier.preferences.get_preference(Woman("w2"))])
                    # print("Rotator's preference list for m1:",
                    #     [x.id for x in self.rotator.preferences.get_preference(Man("m1"))])
                    # print("Rotator's preference list for m2:",
                    #     [x.id for x in self.rotator.preferences.get_preference(Man("m2"))])
                    # print("Rotator's preference list for w1:",
                    #     [x.id for x in self.rotator.preferences.get_preference(Woman("w1"))])
                    # print("Rotator's preference list for w2:",
                    #     [x.id for x in self.rotator.preferences.get_preference(Woman("w2"))])

                    continue
                
                # Get or create child node
                child_node = self._get_or_create_node(new_matching)
                
                # Add edge
                current_node.children.append(child_node)
                child_node.parents.append(current_node)
                
                # Continue BFS if not visited
                if new_matching not in visited:
                    visited.add(new_matching)
                    queue.append(child_node)
        
        return self.man_optimal, self.woman_optimal
    
    def _get_or_create_node(self, matching: Matching) -> LatticeNode:
        """Get existing node or create new one."""
        if matching not in self.nodes:
            self.nodes[matching] = LatticeNode(
                matching=matching,
                rotations=[],
                parents=[],
                children=[]
            )
        return self.nodes[matching]
    
    def get_all_matchings(self) -> List[Matching]:
        """Get all stable matchings in the lattice."""
        return list(self.nodes.keys())
    
    def get_path(self, start: Matching, end: Matching) -> Optional[List[Tuple[Matching, List[Tuple[Man, Woman]]]]]:
        """
        Find a path from start to end matching using rotations.
        
        Returns:
            List of (matching, rotation_used) pairs, or None if no path exists
        """
        if start not in self.nodes or end not in self.nodes:
            return None
        
        # BFS to find path
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path
            
            current_node = self.nodes[current]
            for rotation in current_node.rotations:
                next_matching = self.rotator.perform_rotation(current, rotation)
                if next_matching not in visited:
                    visited.add(next_matching)
                    new_path = path + [(current, rotation)]
                    queue.append((next_matching, new_path))
        
        return None
    
    def print_lattice(self):
        """Print the lattice structure."""
        if not self.nodes:
            print("Lattice not built yet. Call build_lattice() first.")
            return
        
        print("=" * 70)
        print(f"STABLE MATCHING LATTICE: {len(self.nodes)} matchings found")
        print("=" * 70)
        
        # Sort nodes by number of parents (level in lattice)
        nodes_by_level = defaultdict(list)
        for matching, node in self.nodes.items():
            level = len(node.parents)
            nodes_by_level[level].append(node)
        
        for level in sorted(nodes_by_level.keys()):
            print(f"\nLevel {level}:")
            for node in nodes_by_level[level]:
                pairs_str = ", ".join(str(p) for p in sorted(node.matching.pairs, key=lambda x: x.man.id))
                num_rotations = len(node.rotations)
                print(f"  {pairs_str} [{num_rotations} rotation(s) exposed]")
    
    def compare_matchings(self, matching1: Matching, matching2: Matching):
        """Compare two matchings from men's perspective."""
        print(f"\nComparing matchings:")
        print(f"  M1: {matching1}")
        print(f"  M2: {matching2}")
        
        men_better_off = []
        men_worse_off = []
        men_same = []
        
        all_men = set(pair.man for pair in matching1.pairs) | set(matching1.unmatched_men)
        
        for man in all_men:
            partner1 = matching1.get_partner(man)
            partner2 = matching2.get_partner(man)
            
            if partner1 == partner2:
                men_same.append(man)
            elif partner1 is None:
                men_better_off.append(man)  # Going from unmatched to matched
            elif partner2 is None:
                men_worse_off.append(man)  # Going from matched to unmatched
            elif self.preferences.prefers(man, partner2, partner1):
                men_better_off.append(man)
            else:
                men_worse_off.append(man)
        
        print(f"\n  Men better off in M2: {[m.id for m in men_better_off]}")
        print(f"  Men worse off in M2: {[m.id for m in men_worse_off]}")
        print(f"  Men unchanged: {[m.id for m in men_same]}")


# ============================================================================
# COMPREHENSIVE DEMONSTRATION
# ============================================================================

def demo_complete_system():
    """Demonstrate the complete stable matching with rotations system."""
    
    print("=" * 70)
    print("COMPLETE STABLE MATCHING WITH ROTATIONS SYSTEM")
    print("=" * 70)
    
    # Create participants
    m1, m2, m3, m4 = Man("m1"), Man("m2"), Man("m3"), Man("m4")
    w1, w2, w3, w4 = Woman("w1"), Woman("w2"), Woman("w3"), Woman("w4")
    
    # Define preferences (complete lists for this example)
    preferences = PreferenceList({
        m1: [w1, w2, w3, w4],
        m2: [w2, w1, w4, w3],
        m3: [w3, w4, w1, w2],
        m4: [w4, w3, w2, w1],
        w1: [m4, m3, m2, m1],
        w2: [m3, m4, m1, m2],
        w3: [m2, m1, m4, m3],
        w4: [m1, m2, m3, m4]
    })
    
    # 1. Find matchings using Gale-Shapley
    print("\n" + "=" * 70)
    print("1. GALE-SHAPLEY ALGORITHM")
    print("=" * 70)
    
    gs = GaleShapley(preferences)
    man_opt, woman_opt = gs.find_both_optimal()
    
    print(f"\nMan-optimal matching: {man_opt}")
    print(f"Woman-optimal matching: {woman_opt}")
    
    # 2. Verify stability
    print("\n" + "=" * 70)
    print("2. STABILITY VERIFICATION")
    print("=" * 70)
    
    verifier = StabilityVerifier(preferences)
    
    for name, matching in [("Man-optimal", man_opt), ("Woman-optimal", woman_opt)]:
        is_stable, reason, blocking = verifier.is_stable(matching)
        print(f"\n{name}: {'✓ Stable' if is_stable else f'✗ Unstable: {reason}'}")
    
    # Test with an unstable matching
    unstable_dict = {m1: w2, m2: w1, m3: w4, m4: w3}
    all_men = {m1, m2, m3, m4}
    all_women = {w1, w2, w3, w4}
    unstable_matching = Matching.from_dict(unstable_dict, all_men, all_women)
    is_stable, reason, blocking = verifier.is_stable(unstable_matching)
    print(f"\nUnstable test: {'✓ Stable' if is_stable else f'✗ Unstable: {reason}'}")
    if blocking:
        print(f"  Blocking pair: ({blocking[0].id}, {blocking[1].id})")
    
    # 3. Find and perform rotations
    print("\n" + "=" * 70)
    print("3. ROTATION OPERATIONS")
    print("=" * 70)
    
    rotator = RotationOperator(preferences)
    rotations = rotator.find_exposed_rotations(man_opt)
    
    print(f"\nFound {len(rotations)} exposed rotation(s) from man-optimal matching:")
    for i, rotation in enumerate(rotations, 1):
        print(f"\nRotation {i}:")
        cycle_str = " → ".join([f"({m.id},{w.id})" for m, w in rotation])
        print(f"  Cycle: {cycle_str} → (back to start)")
        print(f"  Effect:")
        print(rotator.get_rotation_effect(rotation))
        
        # Perform the rotation
        new_matching = rotator.perform_rotation(man_opt, rotation)
        print(f"  Result: {new_matching}")
        
        # Verify it's stable
        is_stable, _, _ = verifier.is_stable(new_matching)
        print(f"  Stability: {'✓' if is_stable else '✗'}")
    
    # 4. Build complete lattice
    print("\n" + "=" * 70)
    print("4. STABLE MATCHING LATTICE")
    print("=" * 70)
    
    lattice = StableMatchingLattice(preferences)
    man_node, woman_node = lattice.build_lattice()
    
    lattice.print_lattice()
    
    # 5. Demonstrate unequal numbers
    print("\n" + "=" * 70)
    print("5. UNEQUAL NUMBERS DEMONSTRATION")
    print("=" * 70)
    
    m1u, m2u, m3u = Man("m1"), Man("m2"), Man("m3")  # 3 men
    w1u, w2u = Woman("w1"), Woman("w2")  # 2 women
    
    unequal_prefs = PreferenceList({
        m1u: [w1u, w2u],
        m2u: [w2u, w1u],
        m3u: [w1u],  # Only interested in w1
        w1u: [m2u, m1u, m3u],
        w2u: [m1u, m2u]
    })
    
    gs_unequal = GaleShapley(unequal_prefs)
    unequal_man_opt, unequal_woman_opt = gs_unequal.find_both_optimal()
    
    print(f"\nMan-optimal (3 men, 2 women): {unequal_man_opt}")
    print(f"Woman-optimal (3 men, 2 women): {unequal_woman_opt}")
    
    # Verify stability
    verifier_unequal = StabilityVerifier(unequal_prefs)
    is_stable, _, _ = verifier_unequal.is_stable(unequal_man_opt)
    print(f"Man-optimal stable: {'✓' if is_stable else '✗'}")
    
    # Try to find rotations with unequal numbers
    rotator_unequal = RotationOperator(unequal_prefs)
    rotations_unequal = rotator_unequal.find_exposed_rotations(unequal_man_opt)
    print(f"Rotations from man-optimal: {len(rotations_unequal)}")
    
    if rotations_unequal:
        for rotation in rotations_unequal:
            new_matching = rotator_unequal.perform_rotation(unequal_man_opt, rotation)
            print(f"  After rotation: {new_matching}")
    else:
        print("  No rotations found (may be woman-optimal or only stable matching)")
    
    # Build lattice for unequal case
    lattice_unequal = StableMatchingLattice(unequal_prefs)
    lattice_unequal.build_lattice()
    lattice_unequal.print_lattice()


if __name__ == "__main__":
    demo_complete_system()