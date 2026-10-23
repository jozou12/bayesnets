"""One example test per question, with inputs and independently calculated answers.

Run all examples with `python test_q1to4.py`, or one with `-q q2`, for example.
These examples help with debugging; they do not assign grades or cover every
grading case. Q4 checks the posterior, not the required join/eliminate sequence.
The names in Q2-Q4 are arbitrary labels, not names to hard-code in your answers.

To experiment, make a separate copy of this script and keep the supplied version
for submission. When changing an example, keep variable names, factor scopes,
domains, and assignments consistent. Use only values listed in the domains.
For Q4's CPTs, probabilities must sum to one for each parent assignment.
Recalculate expected results independently when inputs change; do not replace
them with your implementation's output or remove checks to obtain a PASS.
"""

import argparse
import math
import sys

import bayesNet as bn
import factorOperations as fo
import hunters
import inference
import layout


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_factor(factor, unconditioned, conditioned, domains, rows):
    require(factor is not None, "Expected a factor; received None.")
    require(factor.unconditionedVariables() == set(unconditioned),
            "Incorrect unconditioned variables.")
    require(factor.conditionedVariables() == set(conditioned),
            "Incorrect conditioned variables.")
    require(factor.variableDomainsDict() == domains, "Incorrect variable domains.")
    for assignment, expected in rows:
        actual = factor.getProbability(assignment)
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
                f"For {assignment}: expected {expected:.10g}, received {actual:.10g}.")


def q1():
    """Check the Ghostbusters graph and domains on a 4-by-3 grid.

    Positions include every cell, including walls. Observation values are
    integers 0 through 12 for this board and the supplied noise bound of 7.
    If changing the board, update position ranges/counts and observation bounds;
    the required Ghostbusters variable names and edges stay the same.
    """
    state = hunters.GameState()
    # This layout supplies dimensions only; the test does not play the game.
    state.initialize(layout.Layout(["%%%%", "%P %", "%%%%"]), 0)
    net = inference.constructBayesNet(state)
    variables = {"Pacman", "Ghost0", "Ghost1", "Observation0", "Observation1"}
    require(net.variablesSet() == variables, "Incorrect Bayes net variables.")
    expected_edges = {("Pacman", "Observation0"), ("Ghost0", "Observation0"),
                      ("Pacman", "Observation1"), ("Ghost1", "Observation1")}
    edges = {(parent, child) for child, parents in net.inEdges().items() for parent in parents}
    require(edges == expected_edges, f"Expected edges {expected_edges}; received {edges}.")
    domains = net.variableDomainsDict()
    positions = {(x, y) for x in range(4) for y in range(3)}
    for variable in ("Pacman", "Ghost0", "Ghost1"):
        require(set(domains[variable]) == positions and len(domains[variable]) == 12,
                f"Incorrect position domain for {variable}; include walls.")
    for variable in ("Observation0", "Observation1"):
        require(set(domains[variable]) == set(range(13)) and len(domains[variable]) == 13,
                f"Incorrect distance domain for {variable}; expected integers 0 through 12.")


def q2():
    """Join a badge-color prior with an alarm-state distribution given the badge.

    Badge is green or gold; Alarm is off or on. Joining P(Badge) and
    P(Alarm | Badge) should give P(Badge, Alarm). Unused is deliberately absent
    from both factors but must remain in the returned domain dictionary.
    """
    domains = {"Badge": ["green", "gold"], "Alarm": ["off", "on"], "Unused": [0, 1]}
    badge = bn.Factor(["Badge"], [], domains)
    badge.setProbability({"Badge": "green"}, 0.4)
    badge.setProbability({"Badge": "gold"}, 0.6)
    alarm = bn.Factor(["Alarm"], ["Badge"], domains)
    # Each row is (badge color, alarm state, conditional probability).
    inputs = [("green", "off", 0.15), ("green", "on", 0.85),
              ("gold", "off", 0.8), ("gold", "on", 0.2)]
    for badge_value, alarm_value, probability in inputs:
        alarm.setProbability({"Badge": badge_value, "Alarm": alarm_value}, probability)
    result = fo.joinFactors([badge, alarm])
    # Each expected joint entry is a product, e.g. green/off: 0.4 * 0.15 = 0.06.
    # If inputs change, recompute these values and update the scopes/domains below.
    expected = [("green", "off", 0.06), ("green", "on", 0.34),
                ("gold", "off", 0.48), ("gold", "on", 0.12)]
    check_factor(result, {"Badge", "Alarm"}, set(), domains,
                 [({"Badge": b, "Alarm": a}, p) for b, a, p in expected])


def q3():
    """Sum out Alarm from badge/alarm probabilities conditioned on a zone.

    Zone is quiet or busy. The input is P(Badge, Alarm | Zone); the result should
    be P(Badge | Zone), keeping Zone conditioned and preserving Unused's domain.
    For each zone and badge color, add the off and on entries for Alarm.
    """
    domains = {"Badge": ["green", "gold"], "Alarm": ["off", "on"],
               "Zone": ["quiet", "busy"], "Unused": [0, 1]}
    joint = bn.Factor(["Badge", "Alarm"], ["Zone"], domains)
    # Each row is (zone, badge color, alarm state, conditional joint probability).
    inputs = [("quiet", "green", "off", 0.1), ("quiet", "green", "on", 0.25),
              ("quiet", "gold", "off", 0.4), ("quiet", "gold", "on", 0.25),
              ("busy", "green", "off", 0.3), ("busy", "green", "on", 0.25),
              ("busy", "gold", "off", 0.2), ("busy", "gold", "on", 0.25)]
    for zone, badge, alarm, probability in inputs:
        joint.setProbability({"Zone": zone, "Badge": badge, "Alarm": alarm}, probability)
    result = fo.eliminate(joint, "Alarm")
    # Example: quiet/green is 0.1 + 0.25 = 0.35.
    # Changing the input or eliminated variable requires new expected rows/scopes.
    expected = [("quiet", "green", 0.35), ("quiet", "gold", 0.65),
                ("busy", "green", 0.55), ("busy", "gold", 0.45)]
    check_factor(result, {"Badge"}, {"Zone"}, domains,
                 [({"Zone": z, "Badge": b}, p) for z, b, p in expected])


def q4():
    """Infer fuel level from a lit indicator in the chain Fuel -> Sensor -> Lamp.

    Fuel is full or low. A noisy Sensor is on or off, and controls whether the
    Lamp is lit or dark. Query P(Fuel | Lamp=lit), eliminating hidden Sensor.
    This checks the normalized posterior and evidence-restricted domains;
    the TA grading suite separately checks the required operation sequence.
    """
    domains = {"Fuel": ["full", "low"], "Sensor": ["on", "off"], "Lamp": ["lit", "dark"]}
    net = bn.constructEmptyBayesNet(list(domains), [("Fuel", "Sensor"), ("Sensor", "Lamp")], domains)
    prior = bn.Factor(["Fuel"], [], domains)
    prior.setProbability({"Fuel": "full"}, 0.3)
    prior.setProbability({"Fuel": "low"}, 0.7)
    sensor = bn.Factor(["Sensor"], ["Fuel"], domains)
    for fuel, on_probability in [("full", 0.8), ("low", 0.2)]:
        sensor.setProbability({"Fuel": fuel, "Sensor": "on"}, on_probability)
        sensor.setProbability({"Fuel": fuel, "Sensor": "off"}, 1 - on_probability)
    lamp = bn.Factor(["Lamp"], ["Sensor"], domains)
    for sensor_value, lit_probability in [("on", 0.9), ("off", 0.1)]:
        lamp.setProbability({"Sensor": sensor_value, "Lamp": "lit"}, lit_probability)
        lamp.setProbability({"Sensor": sensor_value, "Lamp": "dark"}, 1 - lit_probability)
    for variable, factor in [("Fuel", prior), ("Sensor", sensor), ("Lamp", lamp)]:
        net.setCPT(variable, factor)
    result = inference.inferenceByVariableElimination(net, ["Fuel"], {"Lamp": "lit"}, ["Sensor"])
    # P(full, lit) = .3 * (.8*.9 + .2*.1) = .222; P(low, lit) = .182.
    # Divide by P(lit) = .404 to get the expected posterior entries below.
    # If modifying the model, keep each CPT normalized for every parent setting
    # and update the query, evidence, elimination order, and expected result together.
    evidence_domains = dict(domains, Lamp=["lit"])
    check_factor(result, {"Fuel"}, {"Lamp"}, evidence_domains,
                 [({"Fuel": "full", "Lamp": "lit"}, 111 / 202),
                  ({"Fuel": "low", "Lamp": "lit"}, 91 / 202)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--question", choices=("q1", "q2", "q3", "q4"))
    args = parser.parse_args()
    failures = 0
    for question in ([args.question] if args.question else ["q1", "q2", "q3", "q4"]):
        try:
            globals()[question]()
        except SystemExit as error:
            failures += 1
            print(f"[FAIL] {question}: function exited before returning a result (exit {error.code}).")
        except Exception as error:
            failures += 1
            print(f"[FAIL] {question}: {type(error).__name__}: {error}")
        else:
            print(f"[PASS] {question}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
