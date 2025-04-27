A Bayesian Belief Network is a graphical model representing probabilistic relationships among random variables. It assumes conditional independence among attributes and uses joint probability based on conditions:

P(attribute∣parent).

Applications:
Prediction, anomaly detection, diagnostics, automated insight, reasoning, time series prediction, decision-making under uncertainty.

Components:

Directed Acyclic Graph (DAG) — Nodes (variables) and directed arcs (dependencies).

Conditional Probability Tables (CPT) — Probabilities of each node given its parent(s).

Structure:

Nodes: Represent random variables (discrete or continuous).

Arcs: Represent causal relationships.

Parent Node: A node with a directed arrow pointing to another node (e.g., A → B: A is parent of B).

Independence: No direct link means variables are independent.

Extended Form:
An Influence Diagram is a generalized Bayesian Network for decision-making under uncertainty.

Key Components:

Causal Relationships (structure of the graph).

Actual Probabilities (numerical values in CPT).
