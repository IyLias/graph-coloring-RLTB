import networkx as nx




def readDIMACS_graph(graph_file_path):
    G = nx.Graph()

    with open(graph_file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if line.strip() == '' or line[0] == 'c':
            continue

        parts = line.split()
        tp = parts[0]

        if tp == 'p':
            # problem line: p <format> <nodes> <edges>
            nodes = int(parts[2])
            edges = int(parts[3])
            G.add_nodes_from(range(nodes))
        elif tp =='e':
            # Edge line: e <node1> <node2>
            u = int(parts[1])
            v = int(parts[2])
            G.add_edge(u-1,v-1)

    return G
