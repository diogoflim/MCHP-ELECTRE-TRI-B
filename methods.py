from dataclasses import dataclass, field

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
            # Looks for credibility: a S_r b  <=>  sigma_r(a, b) >= lambda_r
            return self.credibility(a, b, node) >= lambda_r

        c = self.concordance(a, b, node)
        if c < lambda_r: return False

        if self.relation == "O2" and self.discordance_veto(a, b, node):
            return False # O2 adds the binary non-veto test; O1 uses concordance only

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