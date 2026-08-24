from dataclasses import dataclass, field
import math
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class CriterionNode:
    name: str
    children: list["CriterionNode"] = field(default_factory=list)
    leaf_id: str | None = None

    def is_leaf(self):
        return self.leaf_id is not None

    def leaves(self):
        if self.is_leaf():
            return [self.leaf_id]
        leaves_list = []
        for child in self.children:
            leaves_list.extend(child.leaves())
        return leaves_list

@dataclass
class HierarchicalElectreTriB:
    root: CriterionNode
    weights: dict[str, float]
    q: dict[str, float]
    p: dict[str, float]
    v: dict[str, float]
    lambdas: dict[str, float]
    profiles: dict[str, dict[str, float]]
    categories: list[str]
    relation: str = "O2"  # outranking relation: "O1", "O2" or "O3"

    def partial_concordance(self, a_value, b_value, q, p):
        diff = b_value - a_value
        return 1.0 if diff <= q else (0.0 if diff >= p else (p - diff) / (p - q))
   
    def concordance(self, a, b, node: CriterionNode):
        total = 0.0
        for leaf in node.leaves():
            phi = self.partial_concordance(a[leaf], b[leaf], self.q[leaf], self.p[leaf])
            total += self.weights[leaf] * phi
        return total

    def discordance_veto(self, a, b, node: CriterionNode):
        for leaf in node.leaves():
            if b[leaf] - a[leaf] >= self.v[leaf]:
                return True
        return False

    def partial_discordance(self, a_value, b_value, p, v):
        diff = b_value - a_value
        return 0.0 if diff <= p else (1.0 if diff >= v else (diff - p) / (v - p))
        

    def credibility(self, a, b, node: CriterionNode):
        c = self.concordance(a, b, node)
        sigma = c
        for leaf in node.leaves():
            d = self.partial_discordance(a[leaf], b[leaf], self.p[leaf], self.v[leaf])
            if d > c: 
                sigma *= (1.0 - d) / (1.0 - c)
        return sigma

    def outranks(self, a, b, node: CriterionNode):
        lambda_r = self.lambdas[node.name]

        if self.relation == "O3":
            return self.credibility(a, b, node) >= lambda_r

        c = self.concordance(a, b, node)
        if c < lambda_r: return False

        if self.relation == "O2" and self.discordance_veto(a, b, node):
            return False 

        return True

    def assign_pessimistic(self, alternative, node: CriterionNode):
        # compare from best boundary to worst boundary
        # alternative should outrank profile
        for h in reversed(range(1, len(self.categories))):
            profile = self.profiles[f"b{h}"]
            if self.outranks(alternative, profile, node):
                return self.categories[h]
        return self.categories[0]

    def assign_optimistic(self, alternative, node: CriterionNode):
        # compare from worst boundary to best boundary
        # profile should outrank alternative
        for h in range(1, len(self.categories)):
            profile = self.profiles[f"b{h}"]
            if self.outranks(profile, alternative, node) and not self.outranks( alternative, profile, node):
                return self.categories[h - 1]
        return self.categories[-1]

    def _category_index(self, category):
        return self.categories.index(category)

    def assign_optimistic_modified(self, alternative, node: CriterionNode):
        """
        Modified optimistic procedure ensuring the category of a node is not 
        lower than the minimum category of its direct subcriteria.
            k' = max( k_node, min_j k_child_j )
        """
        k_node = self._category_index(self.assign_optimistic(alternative, node))

        if node.is_leaf() or all(child.is_leaf() for child in node.children):
            return self.categories[k_node]

        child_indices = [
            self._category_index(self.assign_optimistic_modified(alternative, child))
            for child in node.children
            if not child.is_leaf()
        ]
        
        k = max(k_node, min(child_indices)) if child_indices else k_node
        
        return self.categories[k]

    def assign_all_nodes(self, alternative, procedure="pessimistic"):
        procedures = {
            "pessimistic": self.assign_pessimistic,
            "optimistic": self.assign_optimistic,
            "optimistic_modified": self.assign_optimistic_modified}
        
        if procedure not in procedures:
            raise ValueError(f"procedure must be one of {list(procedures)}, got {procedure!r}")
        
        assign = procedures[procedure]
        results = {}
        def visit(node):
            if not node.is_leaf():
                results[node.name] = assign(alternative, node)
                for child in node.children:
                    visit(child)

        visit(self.root)
        return results
    


def params_to_df(records):
    """DataFrame with MultiIndex columns (param, id), indexed by iteration."""
    rows = []

    for rec in records:
        row = {}
        for name, d in rec.items():
            for k, val in d.items():
                row[(name, k)] = val
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["param", "id"])
    out.index.name = "iteration"
    return out


def set_equal_weights_topdown(node, weight, out=None):
    """Split `weight` equally among the children and recurse down to the leaves,
    accumulating every weight into the same `out` dict (returned at the root)."""
    if out is None:
        out = {}
    if node.is_leaf():
        out[node.leaf_id] = weight
        return out
    child_weight = weight / len(node.children)
    for child in node.children:
        set_equal_weights_topdown(child, child_weight, out)
    return out


def compute_lambdas(node, weights, lambda_frac, out=None):
    """Compute the lambdas from a weights dict: 
    for each internal node, lambda[node] = lambda_frac * (sum of the weights of the leaves under it).
    """
    if out is None:
        out = {}
    if not node.is_leaf():
        out[node.name] = lambda_frac * sum(weights[lf] for lf in node.leaves())
        for child in node.children:
            compute_lambdas(child, weights, lambda_frac, out)
    return out


def compute_hierarchy(country, root, df, results, show=True):
    """Build (and optionally print) a country's criteria tree"""
    if country not in results.index:
        raise KeyError(f"Country '{country}' not found. "
                       f"e.g.: {list(results.index[:5])} ...")
    values = df.loc[country]
    rows = []

    def visit(node, level):
        if node.is_leaf():
            cls, value = "", float(values[node.leaf_id])
        else:
            cls, value = results.loc[country, node.name], None
        rows.append({"Level": level, "Criterion": node.name,
                     "Class": cls, "Value": value})
        for child in node.children:
            visit(child, level + 1)

    visit(root, 0)
    table = pd.DataFrame(rows)
    if show:
        print(f"Country: {country}\n" + "=" * 62)
        for _, r in table.iterrows():
            label = "  " * r["Level"] + r["Criterion"]
            if r["Class"]:
                print(f"{label:<55s} -> {r['Class']}")
            else:
                print(f"{label:<55s}    (value = {r['Value']:.2f})")
    return table


