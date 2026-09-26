import networkx as nx
import matplotlib.pyplot as plt


def create_imu_mapping(acc_nodes, gyro_nodes):
    # Map names to indices (e.g., 'SCG_X' -> 0)
    # Mapping Info: {'SCG_X': 0, 'SCG_Y': 1, 'SCG_Z': 2, 'GCG_X': 3, 'GCG_Y': 4, 'GCG_Z': 5}
    mapping = {name: i for i, name in enumerate(acc_nodes + gyro_nodes)}
    # Map indices back to names (for visualization)
    reverse_mapping = {i: name for name, i in mapping.items()}
    return mapping, reverse_mapping


def draw_imu_graph(G, reverse_mapping):
    pos = nx.kamada_kawai_layout(G)
    labels = {i: f"$\\mathbf{{{name}}}$" for i, name in reverse_mapping.items()}

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_color="lightblue",
        node_size=4000,
        font_size=24,
        font_weight="bold",
    )

    plt.margins(0.2)
    plt.show()


def imu_graph_relationship_all():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_edges_from([(3, 4), (4, 5), (5, 3)])

    # GyroX-AccY, GyroX-AccZ
    # GyroY-AccX, GyroY-AccZ
    # GyroZ-AccX, GyroZ-AccY
    G.add_edges_from(
        [(0, 3), (1, 4), (2, 5), (3, 1), (3, 2), (4, 0), (4, 2), (5, 0), (5, 1)]
    )

    return G


if __name__ == "__main__":
    acc_list = ["SCG_X", "SCG_Y", "SCG_Z"]
    gyro_list = ["GCG_X", "GCG_Y", "GCG_Z"]

    mapping, rev_mapping = create_imu_mapping(acc_list, gyro_list)

    print("Mapping Info:", mapping)
    G = imu_graph_relationship_all()
    draw_imu_graph(G, rev_mapping)
