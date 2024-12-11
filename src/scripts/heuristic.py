import networkx as nx

class HeuristicBase:
    """Base class for all heuristics."""
    def __init__(self, alpha=0):
        self.alpha = alpha

    def select(self, graph, k):
        """To be implemented by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses")


class DegreeHeuristic(HeuristicBase):
    """Degree heuristic: out_degree + alpha * in_degree."""
    def select(self, graph, k):
        scores = {
            node: graph.out_degree(node) + (self.alpha * graph.in_degree(node))
            for node in graph.nodes()
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]


class DegreeDiscountHeuristic(HeuristicBase):
    """Degree discount heuristic adjusted with alpha."""
    def select(self, graph, k):
        degrees = {
            node: graph.out_degree(node) + (self.alpha * graph.in_degree(node))
            for node in graph.nodes()
        }
        t = {node: 0 for node in graph.nodes()}  # Tracks selected neighbors
        discounted_degrees = degrees.copy()
        selected_nodes = []

        for _ in range(k):
            u = max(discounted_degrees, key=discounted_degrees.get)
            selected_nodes.append(u)

            for neighbor in graph.neighbors(u):
                if neighbor not in selected_nodes:
                    t[neighbor] += 1
                    discounted_degrees[neighbor] = degrees[neighbor] - t[neighbor]

            discounted_degrees.pop(u)

        return selected_nodes


class TwoStepPressureHeuristic(HeuristicBase):
    """Two-step heuristic for influence maximization."""
    def select(self, graph, k):
        top_out_degree_nodes = sorted(
            graph.nodes(),
            key=lambda node: graph.out_degree(node),
            reverse=True
        )[:5 * k]

        refined_nodes = sorted(
            top_out_degree_nodes,
            key=lambda node: sum(
                graph[neighbor][node].get('weight', 1) for neighbor in graph.predecessors(node)
            ),
            reverse=True
        )[:k]

        return refined_nodes


class PressureFlowHeuristic(HeuristicBase):
    """Pressure flow heuristic: in-pressure * out-degree."""
    def select(self, graph, k):
        scores = {}
        for node in graph.nodes():
            in_pressure = sum(graph[neighbor][node].get('weight', 1) for neighbor in graph.predecessors(node))
            scores[node] = in_pressure * graph.out_degree(node)

        return sorted(scores, key=scores.get, reverse=True)[:k]


class AmplifiedCoverageHeuristic(HeuristicBase):
    """Amplified coverage heuristic."""
    def select(self, graph, k):
        scores = {}
        for node in graph.nodes():
            neighbor_influence = sum(graph.in_degree(neighbor) for neighbor in graph.neighbors(node))
            scores[node] = graph.out_degree(node) + self.alpha * neighbor_influence

        return sorted(scores, key=scores.get, reverse=True)[:k]


class PressureDegreeDiscountHeuristic(HeuristicBase):
    """Degree discount heuristic adjusted for pressure dynamics."""
    def select(self, graph, k):
        degrees = {node: graph.degree(node) for node in graph.nodes()}
        t = {node: 0 for node in graph.nodes()}
        scores = {}

        for node in graph.nodes():
            in_pressure = sum(graph[neighbor][node].get('weight', 1) for neighbor in graph.predecessors(node))
            scores[node] = degrees[node] + self.alpha * in_pressure

        selected_nodes = []
        for _ in range(k):
            u = max(scores, key=scores.get)
            selected_nodes.append(u)

            for neighbor in graph.neighbors(u):
                if neighbor not in selected_nodes:
                    t[neighbor] += 1
                    scores[neighbor] = degrees[neighbor] - t[neighbor] + self.alpha * in_pressure

            scores.pop(u)

        return selected_nodes


class AdaptiveCombinedHeuristic(HeuristicBase):
    """
    Combines degree, degree discount, and pressure-based influence dynamically.
    """
    def __init__(self, alpha=0, weights=(1, 1, 1)):
        super().__init__(alpha)
        self.weights = weights

    def select(self, graph, k):
        w1, w2, w3 = self.weights
        degrees = {node: graph.degree(node) for node in graph.nodes()}
        t = {node: 0 for node in graph.nodes()}  # Tracks neighbors of selected nodes
        combined_scores = {}

        for node in graph.nodes():
            discounted_degree = degrees[node] - t[node]
            pressure_adjustment = w3 * (self.alpha * graph.in_degree(node))
            combined_scores[node] = w1 * degrees[node] + w2 * discounted_degree + pressure_adjustment

        return sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]


class LocalPressureHeuristic(HeuristicBase):
    """
    Local pressure heuristic:
    degree(node) + alpha * sum(in_degree(neighbors)).
    """
    def select(self, graph, k):
        scores = {}
        for node in graph.nodes():
            neighbor_influence = sum(graph.in_degree(neighbor) for neighbor in graph.neighbors(node))
            scores[node] = graph.degree(node) + self.alpha * neighbor_influence

        return sorted(scores, key=scores.get, reverse=True)[:k]


class HybridDegreeHeuristic(HeuristicBase):
    """
    Hybrid heuristic:
    out_degree(node) * (1 + alpha * in_degree(node)).
    """
    def select(self, graph, k):
        scores = {
            node: graph.out_degree(node) * (1 + self.alpha * graph.in_degree(node))
            for node in graph.nodes()
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]


class AdjustedDegreeDiscountHeuristic(HeuristicBase):
    """
    Degree discount heuristic adjusted with alpha:
    out_degree(node) - selected_neighbors + alpha * in_degree(node).
    """
    def select(self, graph, k):
        degrees = {
            node: graph.out_degree(node) + self.alpha * graph.in_degree(node)
            for node in graph.nodes()
        }
        t = {node: 0 for node in graph.nodes()}  # Tracks number of selected neighbors
        discounted_degrees = degrees.copy()
        selected_nodes = []

        for _ in range(k):
            u = max(discounted_degrees, key=discounted_degrees.get)
            selected_nodes.append(u)

            for neighbor in graph.neighbors(u):
                if neighbor not in selected_nodes:
                    t[neighbor] += 1
                    discounted_degrees[neighbor] = degrees[neighbor] - t[neighbor]

            discounted_degrees.pop(u)

        return selected_nodes
