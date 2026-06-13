"""
topology.py
===========
K-port switch model for limited-connectivity scheduling (Cases 4 & 5).

In a distributed quantum computing architecture, N ELUs are connected through
one or more reconfigurable photonic switches, each with **k ports**.  Each Tock,
each switch independently connects any k of the N total ELUs, giving those k
full all-to-all connectivity.  ELUs on the **same** switch can interact; ELUs
on **different** switches cannot.

There is no fixed wiring graph and no SWAP routing.  The scheduling challenge
is to **bin-pack** operations into Tocks where each switch uses at most k ELU
ports (data + ancilla combined).

Parameters
----------
  k          — ports per switch  (simultaneously connected ELUs per switch)
  n_switches — number of parallel switches  (default 1)
  n_elus     — total ELUs in the system  (must be >= |V| of the logical graph)

Cost model
----------
  CZ gate  : each CZ(u,v) occupies 2 ports per step  (1 data + 1 ancilla)
             on ONE switch.  Max CZ gates per sub-batch per switch = ⌊k/2⌋.
             With S switches: max CZ per sub-batch = S × ⌊k/2⌋.

  MPP meas : a stabilizer generator g_v has support {v} ∪ N(v).
             All support qubits must be on the SAME switch simultaneously,
             so |support| ≤ k is required.  Non-conflicting generators can
             be assigned to different switches within the same Tock.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KPortSwitch:
    """Reconfigurable switch configuration.

    Parameters
    ----------
    k          : number of ports per switch (simultaneously connected ELUs)
    n_elus     : total number of ELUs available (data + ancilla pool)
    n_switches : number of parallel switches (default 1)
    """

    k: int
    n_elus: int
    n_switches: int = 1

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("Switch must have at least 2 ports.")
        if self.n_switches < 1:
            raise ValueError("Must have at least 1 switch.")
        if self.k > self.n_elus:
            # Can't connect more ELUs than exist — clamp
            self.k = self.n_elus

    # ── Queries ───────────────────────────────────────────────────────────

    def max_cz_per_step(self) -> int:
        """Max CZ gates per sub-batch across ALL switches.

        Each switch handles ⌊k/2⌋ CZ gates.
        With S switches: S × ⌊k/2⌋ total.
        """
        return self.n_switches * (self.k // 2)

    def max_cz_per_switch(self) -> int:
        """Max CZ gates per sub-batch on a SINGLE switch."""
        return self.k // 2

    def can_fit_support(self, support_size: int) -> bool:
        """Check if a single MPP measurement's support fits in one switch.

        Port cost = support_size (data) + 1 (ancilla ELU).
        """
        return (support_size + 1) <= self.k

    @property
    def total_ports(self) -> int:
        """Total ports across all switches."""
        return self.n_switches * self.k

    @property
    def is_unlimited(self) -> bool:
        """True when a single switch can connect all ELUs (= Cases 1-3)."""
        return self.k >= self.n_elus

    # ── Pretty print ─────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if self.n_switches == 1:
            return f"KPortSwitch(k={self.k}, n_elus={self.n_elus})"
        return (f"KPortSwitch(k={self.k}, n_switches={self.n_switches}, "
                f"n_elus={self.n_elus})")
