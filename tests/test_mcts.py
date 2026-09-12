"""Tests for the MCTS discrete action set and planner configuration validation.

CPU-only. `planners.mcts` imports neither `isaacgym` nor `gym`, so these tests run
without a GPU by constructing `MCTSSolver` against a minimal stub environment.

The stub exposes only `.num_envs` and `.device`. That is deliberate: per
`AGENTS.md`, planners may reach the environment only through `generate(...)`,
`compute_heuristic_values(...)`, and the spaces/attributes the environment
exposes. If `MCTSSolver` ever starts touching `env.sim` or other simulator
internals, these tests fail loudly rather than silently crossing the layer
boundary.

Scope: Tier 1 only — the shipped action set, action-grid construction, and
configuration validation. Batched expansion and padding behavior need a recording
generative stub and are planned as Tier 2.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from planners.mcts import MCTSSolver

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANNERS_CONFIG = REPO_ROOT / "configs" / "planners.yaml"
ENV_CONFIG = REPO_ROOT / "configs" / "config.yaml"

#: Key under `mcts:` that holds the explicit action list.
MCTS_ACTIONS_KEY = "actions"

#: Keys removed by this change. Retained here only so the tests can assert their
#: absence; there is no backward-compatible read path.
LEGACY_ACTION_KEYS = ("v_vals", "omega_vals")

# ---------------------------------------------------------------------------
# Assumptions pending confirmation
#
# The two items below encode design choices that were raised for confirmation and
# have not been answered yet. They are isolated here so that flipping either one is
# a one-line change rather than a rewrite.
#
# 1. `CCW_SIGN` / `CW_SIGN`: which normalized angular value is physically
#    counterclockwise. The wheel conversion in `envs/planning_env.py` negates both
#    wheel targets, so the mapping cannot be read off the source. Verify
#    empirically with `scripts/debug_rl.py` (`0 1.0 30` vs `0 -1.0 30`) before
#    trusting these values. The structural test
#    `ActionSetContractTests.test_shipped_action_structure` is sign-independent and
#    will remain meaningful either way.
#
# 2. Action order. An action's index is its identity: it is stored in
#    `node.action_taken`, written to the run log, and used as the tie-break key in
#    `BaseSearcher.search()` (which keeps the first maximum encountered, so lower
#    indices win exact ties). Forward actions are listed first deliberately.
# ---------------------------------------------------------------------------
CCW_SIGN = 1.0
CW_SIGN = -1.0

EXPECTED_ACTIONS = [
    [1.0, 0.0],        # forward, straight
    [1.0, CCW_SIGN],   # forward + counterclockwise
    [1.0, CW_SIGN],    # forward + clockwise
    [0.0, CCW_SIGN],   # rotate counterclockwise in place
    [0.0, CW_SIGN],    # rotate clockwise in place
]

#: The action that must never appear: it would let the robot stand still, which is
#: what the no-progress detector in `scripts/run_mcts.py` exists to catch.
STATIONARY_ACTION = [0.0, 0.0]

#: A stub env large enough that action-grid construction is never limited by batch
#: capacity. The cross-file capacity invariant is asserted separately in
#: `test_action_count_fits_planning_batch`.
GENEROUS_ENV_COUNT = 64

DEFAULT_BASE_SECTION = {
    "c_param": 1.414,
    "num_iterations": 100,
    "max_expansions": 300,
    "gamma": 0.95,
}


class _StubEnv:
    """Minimal stand-in for `RoombaPlanningEnv`.

    Provides exactly the two attributes `BaseSearcher` reads. No simulator, no
    `generate`, no Gym spaces — which is what makes these tests CPU-only and what
    keeps the planner inside its layer.
    """

    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = torch.device(device)


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def shipped_num_planning_envs():
    return load_yaml(ENV_CONFIG)["env"]["num_planning_envs"]


def shipped_mcts_section():
    return load_yaml(PLANNERS_CONFIG)["mcts"]


class SolverTestCase(unittest.TestCase):
    """Shared helpers for building solvers from synthetic or shipped configs."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def write_planners_config(self, mcts=None, base=None, filename="planners.yaml"):
        """Write a synthetic planners config and return its path.

        `mcts=None` omits the `mcts:` section entirely, which is how the
        missing-key case is exercised.
        """
        document = {"base": dict(base) if base is not None else dict(DEFAULT_BASE_SECTION)}
        if mcts is not None:
            document["mcts"] = mcts
        path = self.tmp_path / filename
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        return path

    def make_solver(self, mcts=None, num_envs=GENEROUS_ENV_COUNT, base=None):
        """Build an `MCTSSolver` from a synthetic config."""
        config_path = self.write_planners_config(mcts=mcts, base=base)
        return MCTSSolver(_StubEnv(num_envs), config_path=str(config_path))

    def make_shipped_solver(self, num_envs=None):
        """Build an `MCTSSolver` from the real `configs/planners.yaml`.

        Defaults to a generous batch so that a config/code mismatch in
        `num_planning_envs` cannot mask an action-set assertion.
        """
        if num_envs is None:
            num_envs = GENEROUS_ENV_COUNT
        return MCTSSolver(_StubEnv(num_envs), config_path=str(PLANNERS_CONFIG))

    def action_pairs(self, solver):
        """Solver actions as a plain list of `[v, omega]` lists."""
        return solver.actions.tolist()


class ActionSetContractTests(SolverTestCase):
    """The action set actually shipped in `configs/planners.yaml`."""

    def test_shipped_action_count_is_five(self):
        solver = self.make_shipped_solver()
        self.assertEqual(
            solver.num_actions,
            5,
            f"expected 5 discrete actions, got {solver.num_actions}: {self.action_pairs(solver)}",
        )

    def test_shipped_actions_match_expected_pairs(self):
        """Exact list equality, order included.

        Order matters because the index is the action's identity throughout the
        planner and the run log.
        """
        solver = self.make_shipped_solver()
        self.assertEqual(self.action_pairs(solver), EXPECTED_ACTIONS)

    def test_no_stationary_action_in_shipped_set(self):
        """The stationary action must be absent by design.

        With it present the robot can park indefinitely; the no-progress detector
        in `scripts/run_mcts.py` exists because that was previously possible.
        """
        solver = self.make_shipped_solver()
        pairs = self.action_pairs(solver)
        for pair in pairs:
            self.assertNotEqual(
                pair,
                STATIONARY_ACTION,
                "stationary action (v=0, omega=0) must not be in the action set",
            )

    def test_shipped_action_structure(self):
        """Sign-independent shape of the set.

        Two pure rotations plus three forward actions. This test stays valid
        regardless of which sign convention `CCW_SIGN` resolves to.
        """
        solver = self.make_shipped_solver()
        pairs = self.action_pairs(solver)

        rotations = [pair for pair in pairs if pair[0] == 0.0]
        forward = [pair for pair in pairs if pair[0] != 0.0]

        self.assertEqual(len(rotations), 2, f"expected 2 pure rotations, got {rotations}")
        self.assertEqual(len(forward), 3, f"expected 3 forward actions, got {forward}")

        # Rotations must be non-zero and opposite, or they are not rotations.
        rotation_rates = sorted(pair[1] for pair in rotations)
        self.assertEqual(len(set(rotation_rates)), 2, "the two rotations must differ")
        self.assertEqual(rotation_rates[0], -rotation_rates[1], "rotations must be opposite")

        # Exactly one forward action goes straight, the other two are mirrored.
        straight = [pair for pair in forward if pair[1] == 0.0]
        turning = [pair for pair in forward if pair[1] != 0.0]
        self.assertEqual(len(straight), 1, f"expected 1 straight action, got {straight}")
        self.assertEqual(len(turning), 2, f"expected 2 turning actions, got {turning}")
        turn_rates = sorted(pair[1] for pair in turning)
        self.assertEqual(turn_rates[0], -turn_rates[1], "forward turns must be mirrored")

    def test_shipped_actions_within_normalized_bounds(self):
        """Every component must be inside [-1, 1].

        `RoombaPlanningEnv.generate` clamps with `torch.clamp(actions, -1.0, 1.0)`,
        so an out-of-range value would be silently truncated instead of erroring.
        """
        solver = self.make_shipped_solver()
        for pair in self.action_pairs(solver):
            for component in pair:
                self.assertGreaterEqual(component, -1.0, f"{pair} is below -1.0")
                self.assertLessEqual(component, 1.0, f"{pair} is above 1.0")

    def test_shipped_actions_use_max_velocity(self):
        """The forward actions should be at the velocity limit, not a fraction of it."""
        solver = self.make_shipped_solver()
        forward = [pair for pair in self.action_pairs(solver) if pair[0] != 0.0]
        self.assertTrue(forward, "expected forward actions in the set")
        for pair in forward:
            self.assertEqual(pair[0], 1.0, f"{pair} is not at max linear velocity")

    def test_mcts_section_has_no_legacy_action_keys(self):
        """Guards the 'replace outright' decision: no dual config path.

        If both schemas coexisted, it would be ambiguous which one produced a run's
        action set.
        """
        mcts = shipped_mcts_section()
        for key in LEGACY_ACTION_KEYS:
            self.assertNotIn(
                key,
                mcts,
                f"legacy action key '{key}' must be removed; use '{MCTS_ACTIONS_KEY}'",
            )
        self.assertIn(MCTS_ACTIONS_KEY, mcts, f"mcts.{MCTS_ACTIONS_KEY} must be defined")

    def test_num_planning_envs_is_nine(self):
        self.assertEqual(shipped_num_planning_envs(), 9)

    def test_action_count_fits_planning_batch(self):
        """The cross-file invariant: every action needs a batch slot.

        `MCTSSolver._expand` writes the action grid into the leading
        `[:num_actions]` rows of a `[num_envs, 2]` tensor, so
        `num_actions <= num_planning_envs` must hold. This reads both files
        directly rather than constructing a solver, so it fails for exactly one
        reason and cannot be masked by unrelated validation.
        """
        actions = shipped_mcts_section()[MCTS_ACTIONS_KEY]
        num_planning_envs = shipped_num_planning_envs()
        self.assertLessEqual(
            len(actions),
            num_planning_envs,
            f"{len(actions)} actions do not fit in num_planning_envs={num_planning_envs}",
        )


class ActionGridConstructionTests(SolverTestCase):
    """`_create_action_grid` reads an explicit list from `mcts.actions`."""

    def test_explicit_list_parsed_in_order(self):
        """Action i must end up at index i.

        The index is the identity used in `node.action_taken`, in the run log, and
        as the `BaseSearcher.search()` tie-break. A silent reorder would change
        results without raising anything.
        """
        mcts = {MCTS_ACTIONS_KEY: [[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]]}
        solver = self.make_solver(mcts=mcts)
        self.assertEqual(self.action_pairs(solver), mcts[MCTS_ACTIONS_KEY])

    def test_action_tensor_is_n_by_two(self):
        mcts = {MCTS_ACTIONS_KEY: [[1.0, 0.0], [0.0, 1.0]]}
        solver = self.make_solver(mcts=mcts)
        self.assertEqual(tuple(solver.actions.shape), (2, 2))
        self.assertEqual(solver.num_actions, 2)

    def test_action_tensor_dtype_is_float32(self):
        """`_expand` writes these into a float32 batch tensor."""
        solver = self.make_solver(mcts={MCTS_ACTIONS_KEY: [[1.0, 0.0]]})
        self.assertEqual(solver.actions.dtype, torch.float32)

    def test_action_tensor_on_env_device(self):
        solver = self.make_solver(mcts={MCTS_ACTIONS_KEY: [[1.0, 0.0]]})
        self.assertEqual(solver.actions.device, torch.device("cpu"))

    def test_single_action_grid_is_allowed(self):
        solver = self.make_solver(mcts={MCTS_ACTIONS_KEY: [[0.0, CW_SIGN]]})
        self.assertEqual(solver.num_actions, 1)

    def test_duplicate_actions_are_preserved(self):
        """Duplicates are not de-duplicated.

        If they ever were, `num_actions` would shrink and break the
        actions-versus-batch capacity arithmetic.
        """
        mcts = {MCTS_ACTIONS_KEY: [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}
        solver = self.make_solver(mcts=mcts)
        self.assertEqual(self.action_pairs(solver), mcts[MCTS_ACTIONS_KEY])

    def test_missing_actions_key_fails_fast(self):
        """No fallback default: a typo must error, not silently build a grid.

        The previous implementation fell back to a built-in product grid, which
        meant a misspelled key produced a valid-looking run with the wrong actions.
        """
        solver_factory = lambda: self.make_solver(mcts={"unrelated": 1})
        with self.assertRaises((KeyError, ValueError)):
            solver_factory()

    def test_empty_action_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.make_solver(mcts={MCTS_ACTIONS_KEY: []})

    def test_malformed_entry_raises(self):
        """Wrong arity must not become a 1-component or truncated action."""
        for malformed in ([1.0], [1.0, 0.0, 0.0]):
            with self.subTest(entry=malformed):
                with self.assertRaises((ValueError, TypeError)):
                    self.make_solver(mcts={MCTS_ACTIONS_KEY: [malformed]})

    def test_non_numeric_entry_raises(self):
        with self.assertRaises((ValueError, TypeError)):
            self.make_solver(mcts={MCTS_ACTIONS_KEY: [["forward", 1.0]]})

    def test_out_of_range_action_is_rejected(self):
        """Out-of-range actions are rejected at construction, not clamped later.

        `generate` clamps to [-1, 1], so accepting 1.5 would silently drive at 1.0
        while the config claimed otherwise.

        If this behavior is reverted to accept-and-clamp, invert this test rather
        than deleting it — the leniency should be a recorded decision.
        """
        for out_of_range in ([1.5, 0.0], [-1.5, 0.0], [0.0, 2.0]):
            with self.subTest(entry=out_of_range):
                with self.assertRaises(ValueError):
                    self.make_solver(mcts={MCTS_ACTIONS_KEY: [out_of_range]})


class ValidationInvariantTests(SolverTestCase):
    """`_validate_config` invariants, retested at the new action scale."""

    def test_more_actions_than_envs_raises(self):
        mcts = {MCTS_ACTIONS_KEY: [[1.0, float(i) / 10.0] for i in range(10)]}
        with self.assertRaises(ValueError):
            self.make_solver(mcts=mcts, num_envs=9)

    def test_actions_equal_to_envs_is_allowed(self):
        """Boundary: the guard is `num_actions > num_envs`, so equality must pass."""
        mcts = {MCTS_ACTIONS_KEY: [[1.0, float(i) / 10.0] for i in range(9)]}
        solver = self.make_solver(mcts=mcts, num_envs=9)
        self.assertEqual(solver.num_actions, 9)

    def test_max_expansions_below_action_count_raises(self):
        mcts = {MCTS_ACTIONS_KEY: [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, -1.0], [1.0, -1.0]]}
        base = dict(DEFAULT_BASE_SECTION, max_expansions=3)
        with self.assertRaises(ValueError):
            self.make_solver(mcts=mcts, base=base)

    def test_num_iterations_below_one_raises(self):
        base = dict(DEFAULT_BASE_SECTION, num_iterations=0)
        with self.assertRaises(ValueError):
            self.make_solver(mcts={MCTS_ACTIONS_KEY: [[1.0, 0.0]]}, base=base)

    def test_gamma_out_of_range_raises(self):
        for bad_gamma in (-0.1, 1.1):
            with self.subTest(gamma=bad_gamma):
                base = dict(DEFAULT_BASE_SECTION, gamma=bad_gamma)
                with self.assertRaises(ValueError):
                    self.make_solver(mcts={MCTS_ACTIONS_KEY: [[1.0, 0.0]]}, base=base)

    def test_error_messages_do_not_reference_removed_keys(self):
        """A validation error must not tell the user to edit config that is gone.

        The empty-grid message previously named `v_vals`/`omega_vals`, which no
        longer exist after this change.
        """
        messages = []

        for mcts, base in (
            ({MCTS_ACTIONS_KEY: []}, None),
            ({MCTS_ACTIONS_KEY: [[1.0, float(i) / 10.0] for i in range(10)]}, None),
            ({MCTS_ACTIONS_KEY: [[1.0, 0.0]]}, dict(DEFAULT_BASE_SECTION, max_expansions=0)),
            ({MCTS_ACTIONS_KEY: [[1.0, 0.0]]}, dict(DEFAULT_BASE_SECTION, num_iterations=0)),
            ({MCTS_ACTIONS_KEY: [[1.0, 0.0]]}, dict(DEFAULT_BASE_SECTION, gamma=2.0)),
        ):
            num_envs = 9 if len(mcts[MCTS_ACTIONS_KEY]) > 9 else GENEROUS_ENV_COUNT
            try:
                self.make_solver(mcts=mcts, base=base, num_envs=num_envs)
            except (ValueError, KeyError) as exc:
                messages.append(str(exc))
                continue
            self.fail(f"expected a validation error for mcts={mcts}, base={base}")

        self.assertTrue(messages, "expected at least one validation error message")
        for message in messages:
            for key in LEGACY_ACTION_KEYS:
                self.assertNotIn(
                    key,
                    message,
                    f"validation message references removed key '{key}': {message}",
                )

    def test_shipped_configuration_constructs_cleanly(self):
        """End-to-end: the real configs must satisfy every invariant together."""
        solver = MCTSSolver(
            _StubEnv(shipped_num_planning_envs()),
            config_path=str(PLANNERS_CONFIG),
        )
        self.assertEqual(solver.num_actions, 5)
        self.assertLessEqual(solver.num_actions, solver.num_envs)


if __name__ == "__main__":
    unittest.main()
