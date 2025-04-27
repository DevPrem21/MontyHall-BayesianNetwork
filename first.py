from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

model = DiscreteBayesianNetwork([('Guest', 'Monty'), ('Price', 'Monty')])

cpd_guest = TabularCPD('Guest', 3, [[0.33], [0.33], [0.34]])
cpd_price = TabularCPD('Price', 3, [[0.33], [0.33], [0.34]])


cpd_monty = TabularCPD(
    variable='Monty',
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]
    ],
    evidence=['Guest', 'Price'],
    evidence_card=[3, 3]
)

model.add_cpds(cpd_guest, cpd_price, cpd_monty)
model.check_model()

print('Probability distribution, P(Guest)')
print(cpd_guest)
print()

print('Probability distribution, P(Price)')
print(cpd_price)
print()

print('Joint probability distribution, P(Monty | Guest, Price)')
print(cpd_monty)
print()

plt.figure(figsize=(10, 8))

G = nx.DiGraph()
G.add_nodes_from(model.nodes())
G.add_edges_from(model.edges())
pos = nx.spring_layout(G, k=1, iterations=50)
nx.draw(G, pos, with_labels=True, 
        node_color="lightblue",
        node_size=3000, 
        font_size=16,
        font_weight='bold',
        arrows=True,
        edge_color='gray',
        arrowsize=20)

plt.title("Monty Hall Bayesian Network", pad=20, size=16)

plt.tight_layout()
plt.savefig('Data.png', dpi=300, bbox_inches='tight')
plt.close()

infer = VariableElimination(model)
posterior_probability = infer.query(['Price'], evidence={'Guest': 0, 'Monty': 2})

print('Posterior probability, Guest(0) and Monty(2)')
print(posterior_probability)
print()


