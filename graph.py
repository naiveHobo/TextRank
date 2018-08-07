class Graph:
    """
    Implementation of an undirected graph, based on Pygraph
    """

    WEIGHT_ATTRIBUTE_NAME = "weight"
    DEFAULT_WEIGHT = 0

    LABEL_ATTRIBUTE_NAME = "label"
    DEFAULT_LABEL = ""

    def __init__(self):
        # Metadata about edges
        self.edge_properties = {}    # Mapping: Edge -> Dict mapping, label-> str, wt->num
        self.edge_attr = {}          # Key value pairs: (Edge -> Attributes)
        # Metadata about nodes
        self.node_attr = {}          # Pairing: Node -> Attributes
        self.node_neighbors = {}     # Pairing: Node -> Neighbors

    def has_edge(self, edge):
        """Checks if a given edge exists in the graph"""
        u, v = edge
        return (u, v) in self.edge_properties and (v, u) in self.edge_properties

    def edge_weight(self, edge):
        """Returns the weight of the given edge"""
        return self.get_edge_properties(edge).setdefault(self.WEIGHT_ATTRIBUTE_NAME, self.DEFAULT_WEIGHT)

    def neighbors(self, node):
        """Returns a list of neighbors for a given node"""
        return self.node_neighbors[node]

    def has_node(self, node):
        """Checks if the grpah has a given node"""
        return node in self.node_neighbors

    def add_edge(self, edge, wt=1, label='', attrs=None):
        """Adds an edge to the graph"""
        if not attrs:
            attrs = []
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u].append(v)
            if u != v:
                self.node_neighbors[v].append(u)

            self.add_edge_attributes((u, v), attrs)
            self.set_edge_properties((u, v), label=label, weight=wt)
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node, attrs=None):
        """Adds a node to the graph"""
        if attrs is None:
            attrs = []
        if node not in self.node_neighbors:
            self.node_neighbors[node] = []
            self.node_attr[node] = attrs
        else:
            raise ValueError("Node %s already in graph" % node)

    def nodes(self):
        """Returns a list of nodes in the graph"""
        return list(self.node_neighbors.keys())

    def edges(self):
        """Returns a list of edges in the graph"""
        return [a for a in list(self.edge_properties.keys())]

    def del_node(self, node):
        """Deletes a given node from the graph"""
        for each in list(self.neighbors(node)):
            if each != node:
                self.del_edge((each, node))
        del(self.node_neighbors[node])
        del(self.node_attr[node])

    # Helper methods
    def get_edge_properties(self, edge):
        """Returns the properties of an edge"""
        return self.edge_properties.setdefault(edge, {})

    def add_edge_attributes(self, edge, attrs):
        """Sets multiple edge attributes"""
        for attr in attrs:
            self.add_edge_attribute(edge, attr)

    def add_edge_attribute(self, edge, attr):
        """Sets a single edge attribute"""
        self.edge_attr[edge] = self.edge_attributes(edge) + [attr]

        if edge[0] != edge[1]:
            self.edge_attr[(edge[1], edge[0])] = self.edge_attributes((edge[1], edge[0])) + [attr]

    def edge_attributes(self, edge):
        """Returns edge attributes"""
        try:
            return self.edge_attr[edge]
        except KeyError:
            return []

    def set_edge_properties(self, edge, **properties):
        """Sets edge properties"""
        self.edge_properties.setdefault(edge, {}).update(properties)
        if edge[0] != edge[1]:
            self.edge_properties.setdefault((edge[1], edge[0]), {}).update(properties)

    def del_edge(self, edge):
        """Deletes an edge from the graph"""
        u, v = edge
        self.node_neighbors[u].remove(v)
        self.del_edge_labeling((u, v))
        if u != v:
            self.node_neighbors[v].remove(u)
            self.del_edge_labeling((v, u))

    def del_edge_labeling(self, edge):
        """Deletes the labeling of an edge from the graph"""
        keys = list(edge)
        keys.append(edge[::-1])

        for key in keys:
            for mapping in [self.edge_properties, self.edge_attr]:
                try:
                    del (mapping[key])
                except KeyError:
                    pass
