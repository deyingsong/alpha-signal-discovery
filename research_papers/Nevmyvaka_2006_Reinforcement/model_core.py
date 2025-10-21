"""
Core implementation of [Reinforcement Learning for Optimized Trade Execution].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Protocol
import numpy as np
import logging

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("model_core")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Protocols (interfaces) for pluggable environment & value function
# -----------------------------------------------------------------------------
class Environment(Protocol):
    """
    A pluggable environment describing market/order-book semantics and dynamics.
    """

    def end_of_data(self, t: int) -> bool:
        """Return True if there is no more data to process at time index t."""
        ...

    def transform_order_book(self, t: int) -> Tuple[Any, ...]:
        """
        Transform raw order book (at time t) into a finite representation
        o_1, ..., o_R that the policy uses as state features.
        """
        ...

    def simulate_transition(self, x: "State", a: int) -> "State":
        """
        Given current state x and action a, simulate/advance one step to next state y.
        """
        ...

    def immediate_value(self, x: "State", a: int) -> float:
        """
        Immediate value
        """
        ...

    def actions(self, t: int, i: int) -> Iterable[int]:
        """
        Enumerate valid actions a at (t, i). Typically range(0, L+1),
        """
        ...

    def terminal_value(self, x: "State") -> float:
        """
        Value at terminal time; used when t==0 (or as appropriate).
        """
        ...


class ValueFunction(Protocol):
    """
    Storage and lookup for state-action values c(<t, i, o1...oR>, a).
    """

    def get(self, y: "State") -> float:
        """Return max_a' c(y, a') (i.e., V(y))."""
        ...

    def update(self, x: "State", a: int, q: float) -> None:
        """Set Q(x, a)=q and update V(x)=max_a Q(x, a)."""
        ...

    def argmax_action(self, y: "State", env: Environment) -> Tuple[int, float]:
        """Return (a*, V(y)) where a* = argmax_a Q(y, a)."""
        ...


# -----------------------------------------------------------------------------
# State representation
# -----------------------------------------------------------------------------
class State(NamedTuple):
    t: int              # time index
    i: int              # inventory (or other discrete index)
    obs: Tuple[Any, ...]  # (o1, ..., oR) transformed order-book features

    def as_key(self) -> Tuple[int, int, Tuple[Any, ...]]:
        return (self.t, self.i, self.obs)


# -----------------------------------------------------------------------------
# Default in-memory value function (tabular)
# -----------------------------------------------------------------------------
@dataclass
class TabularValueFunction:
    """
    Stores Q(x, a) and V(x)=max_a Q(x,a).
    """
    q_table: Dict[Tuple, Dict[int, float]]
    v_table: Dict[Tuple, float]

    def __init__(self) -> None:
        self.q_table = {}
        self.v_table = {}

    def _ensure_state(self, x_key: Tuple) -> None:
        if x_key not in self.q_table:
            self.q_table[x_key] = {}
            self.v_table[x_key] = -np.inf

    def get(self, y: State) -> float:
        return self.v_table.get(y.as_key(), 0.0)

    def update(self, x: State, a: int, q: float) -> None:
        k = x.as_key()
        self._ensure_state(k)
        self.q_table[k][a] = q
         
        self.v_table[k] = max(self.q_table[k].values())

    def argmax_action(self, y: State, env: Environment) -> Tuple[int, float]:
        y_key = y.as_key()
         
        if y_key in self.q_table and self.q_table[y_key]:
            a_star = max(self.q_table[y_key], key=self.q_table[y_key].get)
            return a_star, self.v_table[y_key]
         
        best_a, best_q = None, -np.inf
        for a in env.actions(y.t, y.i):
            q = env.immediate_value(y, a)
            if q > best_q:
                best_q, best_a = q, a
        if best_a is None:
            best_a, best_q = 0, 0.0
        return best_a, best_q


# -----------------------------------------------------------------------------
# Core algorithm
# -----------------------------------------------------------------------------
@dataclass
class OptimalStrategyConfig:
    V: int  # target amount of share to sell
    H: float # time horizon
    T: int # maximum value of t (elapsed time)
    I: int # maximum value of i (remaining inventory)
    L: int # number of actions in each state


@dataclass
class OptimalStrategyResult:
    value_function: TabularValueFunction
    policy: Dict[Tuple, int]  # state key -> best action


def run_optimal_strategy(
    env: Environment,
    cfg: OptimalStrategyConfig,
    vf: Optional[TabularValueFunction] = None,
) -> OptimalStrategyResult:
    """
    Implements Q-learning
    """
    vf = vf or TabularValueFunction()

    for t in range(cfg.T, -1, -1):
        logger.info(f"Backward pass at t={t}")

        while not env.end_of_data(t):
            obs = env.transform_order_book(t)  # -> (o1, ..., oR)

            # State-space sweep over inventory (or any discrete index i)
            for i in range(0, cfg.I + 1):
                 
                for a in env.actions(t, i):
                    x = State(t=t, i=i, obs=obs)
                    y = env.simulate_transition(x, a)

                    # Immediate + continuation value (Bellman backup)
                    cim = env.immediate_value(x, a)
                    _, v_y = vf.argmax_action(y, env)
                    q = cim + v_y

                    # Update Q(x,a) and V(x)
                    vf.update(x, a, q)

    # Extract greedy policy
    policy: Dict[Tuple, int] = {}
    for x_key, q_actions in vf.q_table.items():
        best_a = max(q_actions, key=q_actions.get)
        policy[x_key] = best_a

    return OptimalStrategyResult(value_function=vf, policy=policy)


# -----------------------------------------------------------------------------
# Example stub environment 
# -----------------------------------------------------------------------------
class DummyEnv(Environment):
    """
    A tiny example that:
      - Uses a rolling pseudo order-book feature (midprice change sign, spread bucket)
      - Random-walk inventory
      - Immediate value = pnl shock - inventory penalty
    This is ONLY for structure testing; replace with market-specific logic.
    """

    def __init__(self, R: int = 2, rng_seed: int = 0) -> None:
        self.R = R
        self._cursor = 0
        self._data_len = 8  # pretend 8 snapshots per t
        self.rng = np.random.default_rng(rng_seed)

    def end_of_data(self, t: int) -> bool:
        end = self._cursor >= self._data_len
        if end:
            self._cursor = 0  # reset for next t
        return end

    def transform_order_book(self, t: int) -> Tuple[int, int]:
        # Example: (delta_mid_sign, spread_bucket)
        self._cursor += 1
        delta_mid_sign = int(self.rng.integers(-1, 2))
        spread_bucket = int(self.rng.integers(0, 3))
        return (delta_mid_sign, spread_bucket)

    def simulate_transition(self, x: State, a: int) -> State:
        # Inventory moves toward action; observations refresh on next call via transform
        next_i = max(0, x.i + (1 if a > 0 else -1 if a < 0 else 0))
        # y.t is previous time (backward induction keeps t non-increasing)
        return State(t=max(0, x.t - 1), i=next_i, obs=x.obs)

    def immediate_value(self, x: State, a: int) -> float:
        # Toy reward: pnl shock aligned with delta_mid_sign minus inventory penalty
        delta_mid_sign, spread_bucket = x.obs
        shock = (1 + spread_bucket) * delta_mid_sign * (a - 1)  # {-1,0,1} -> position tilt
        penalty = 0.1 * (x.i**2)
        return float(shock - penalty)

    def actions(self, t: int, i: int) -> Iterable[int]:
        # Example discrete actions: {-1, 0, +1}, clipped by a max inventory (here I implicit)
        return (-1, 0, +1)

    def terminal_value(self, x: State) -> float:
        return 0.0


# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = OptimalStrategyConfig(V=0, H=0, T=5, I=3, L=1)
    env = DummyEnv()
    result = run_optimal_strategy(env, cfg)

 
    print("\nLearned policy (state_key -> action):")
    sample = 0
    for k, a in result.policy.items():
        print(f"{k} -> {a}")
        sample += 1
        if sample >= 10:
            break
