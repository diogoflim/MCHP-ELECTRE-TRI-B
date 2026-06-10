import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from methods import (CriterionNode, HierarchicalElectreTriB, set_equal_weights_topdown,
                     class_acceptability_smaa, plot_country_pies_smaa, plot_acceptability_smaa,
                     params_to_df, country_hierarchy, compute_lambdas)


# Data loading
def load_data():
    raw = pd.read_csv("Freedom_and_Prosperity.csv", decimal=",")
    print(f"Raw data: {raw.shape[0]} rows x {raw.shape[1]} columns")

    leaf_cols = [
        # Economic subindex
        "Proper_Rights", "Trade_Freedom", "Investm_Freedom", "Womens_Economic_Freedom",
        # Political subindex
        "PF_Elections", "Civil_Liberties", "Political_Rights", "Legislative_Constraints_on_the_Executive",
        # Legal subindex
        "Clarity_of_the_Law", "Judicial_Independence_and_Effectiveness", "Bureaucracy_and_Corruption",
        "Security", "Informality",
        # Prosperity
        "Income", "Environment", "Minorities", "Health", "Education", "Inequality",
    ]

    df = raw[(raw["Index_Year"] == 2024) & (raw["ISO3"].notna())].copy()
    print(f"Countries in 2024: {df.shape[0]}")
    df = df.dropna(subset=leaf_cols)
    print(f"Complete countries (no missing values): {df.shape[0]}")
    df = df.set_index("Name")[leaf_cols]
    print(f"\n{df.shape[0]} countries x {df.shape[1]} criteria")
    print(df.head())
    return df

# Criteria hierarchy
def build_hierarchy():
    Economic = CriterionNode("Economic_Subindex",
                             children=[
                                 CriterionNode("Proper_Rights", leaf_id="Proper_Rights"),
                                 CriterionNode("Trade_Freedom", leaf_id="Trade_Freedom"),
                                 CriterionNode("Investm_Freedom", leaf_id="Investm_Freedom"),
                                 CriterionNode("Womens_Economic_Freedom", leaf_id="Womens_Economic_Freedom"),
                             ])

    Political = CriterionNode("Political_Subindex",
                              children=[
                                  CriterionNode("PF_Elections", leaf_id="PF_Elections"),
                                  CriterionNode("Civil_Liberties", leaf_id="Civil_Liberties"),
                                  CriterionNode("Political_Rights", leaf_id="Political_Rights"),
                                  CriterionNode("Legislative_Constraints_on_the_Executive",
                                                leaf_id="Legislative_Constraints_on_the_Executive"),
                              ])

    Legal = CriterionNode("Legal_Subindex",
                          children=[
                              CriterionNode("Clarity_of_the_Law", leaf_id="Clarity_of_the_Law"),
                              CriterionNode("Judicial_Independence_and_Effectiveness",
                                            leaf_id="Judicial_Independence_and_Effectiveness"),
                              CriterionNode("Bureaucracy_and_Corruption", leaf_id="Bureaucracy_and_Corruption"),
                              CriterionNode("Security", leaf_id="Security"),
                              CriterionNode("Informality", leaf_id="Informality"),
                          ])

    Freedom = CriterionNode("Freedom",
                            children=[Economic, Political, Legal])

    Prosperity = CriterionNode("Prosperity",
                               children=[
                                   CriterionNode("Income", leaf_id="Income"),
                                   CriterionNode("Environment", leaf_id="Environment"),
                                   CriterionNode("Minorities", leaf_id="Minorities"),
                                   CriterionNode("Health", leaf_id="Health"),
                                   CriterionNode("Education", leaf_id="Education"),
                                   CriterionNode("Inequality", leaf_id="Inequality"),
                               ])

    Root = CriterionNode("Freedom_and_Prosperity",
                         children=[Freedom, Prosperity])
    return Root, Freedom, Prosperity, Economic, Political, Legal


# Acceptability
def build_acceptability(counts, df, nodes, categories):
    accept = {}
    for name in df.index:
        row = {}
        for n in nodes:
            total = sum(counts[name][n].values())
            for c in categories:
                row[(n, c)] = counts[name][n][c] / total
        accept[name] = row

    acceptability = pd.DataFrame(accept).T
    acceptability.columns = pd.MultiIndex.from_tuples(acceptability.columns, names=["node", "class"])
    acceptability = acceptability[[(n, c) for n in nodes for c in categories]]
    return acceptability


def sample_weights(node, weight, rng, out=None):
    """Split `weight` among the children with Dirichlet(1,...,1) and recurse down to the leaves.
    Preserves the hierarchy: on average it keeps each dimension's expected weight (Freedom≈0.5, etc.)."""
    if out is None:
        out = {}
    if node.is_leaf():
        out[node.leaf_id] = weight
        return out
    fracs = rng.dirichlet(np.ones(len(node.children)))
    for child, f in zip(node.children, fracs):
        sample_weights(child, weight * f, rng, out)
    return out


def main():
    df = load_data()
    # Hierarchy 
    Root, Freedom, Prosperity, Economic, Political, Legal = build_hierarchy()
    leaves = Root.leaves()
    print(f"\n{len(leaves)} elementary criteria (leaves):")
    print(leaves)

    # Base-case
    weights = set_equal_weights_topdown(Root, 1.0)
    print("\nweights per leaf:")
    for lf in leaves:
        print(f"  {lf:42s} w = {weights[lf]:.4f}")
    print(f"\nsum Freedom    = {sum(weights[lf] for lf in Freedom.leaves()):.4f}")
    print(f"sum Prosperity = {sum(weights[lf] for lf in Prosperity.leaves()):.4f}")
    for sub in (Economic, Political, Legal):
        print(f"sum {sub.name:18s} = {sum(weights[lf] for lf in sub.leaves()):.4f}")

    # Thresholds, lambdas, and profiles
    q = {lf: 4.0 for lf in leaves}
    p = {lf: 6.0 for lf in leaves}
    v = {lf: 40.0 for lf in leaves}
    lambda_frac = 0.75
    lambdas = compute_lambdas(Root, weights, lambda_frac)
    print("\nlambdas per node:")
    for name, lam in lambdas.items():
        print(f"  {name:42s} lambda = {lam:.4f}")
    categories = ["Class_1", "Class_2", "Class_3", "Class_4"]  
    b1 = {lf: 40.0 for lf in leaves}  
    b2 = {lf: 60.0 for lf in leaves}  
    b3 = {lf: 80.0 for lf in leaves}  
    profiles = {"b1": b1, "b2": b2, "b3": b3}

    # Instantiate the Model 
    model = HierarchicalElectreTriB(
        root=Root,
        weights=weights,
        q=q,
        p=p,
        v=v,
        lambdas=lambdas,
        profiles=profiles,
        categories=categories,
        relation="O3")

    nodes = list(lambdas.keys())  
    colors = {"Class_1": "#c0392b", "Class_2": "#f39c12", "Class_3": "#ddfa1f", "Class_4": "#27ae60"}

    # Classification (pessimistic)
    def classify(procedure):
        alternatives_dict = {}
        for alternative_name in df.index:
            alt = df.loc[alternative_name].to_dict()
            alternatives_dict[alternative_name] = model.assign_all_nodes(alt, procedure=procedure)
        return pd.DataFrame(alternatives_dict).T

    res_pess = classify("pessimistic")
    res_pess = res_pess[nodes]
    print("\nBase-case classification (pessimistic):")
    print(res_pess)

    # global-class bar chart
    class_counts = res_pess["Freedom_and_Prosperity"].value_counts().reindex(categories, fill_value=0)
    print("\nDistribution in the global class (Freedom_and_Prosperity):")
    print(class_counts)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(class_counts.index, class_counts.values, color=[colors[c] for c in class_counts.index])
    ax.set_ylabel("number of countries")
    ax.set_title("Hierarchical ELECTRE Tri — global class (pessimistic)")
    for x, y in zip(class_counts.index, class_counts.values):
        ax.text(x, y, str(y), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # per-node bar charts
    ncols = 3
    nrows = math.ceil(len(nodes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()
    for ax, node in zip(axes, nodes):
        node_counts = res_pess[node].value_counts().reindex(categories, fill_value=0)
        ax.bar(node_counts.index, node_counts.values, color=[colors[c] for c in node_counts.index])
        ax.set_title(node)
        ax.set_ylabel("number of countries")
        ax.set_ylim(0, len(df))
        for x, y in zip(node_counts.index, node_counts.values):
            ax.text(x, y, str(y), ha="center", va="bottom")
    for ax in axes[len(nodes):]:
        ax.axis("off")
    fig.suptitle("Hierarchical ELECTRE TRI-B  — class distribution (pessimistic)", y=1.02)
    plt.tight_layout()
    plt.show()

    # country drill-down
    country_hierarchy("Brazil", Root, df, res_pess)

    alt_dicts = {name: df.loc[name].to_dict() for name in df.index}

    
    
    # SMAA-TRI — fixed weights
    print("\n" + "=" * 70)
    print("SMAA-TRI — fixed weights")
    print("=" * 70)
    N_SIM = 10000
    SEED = 42
    rng = np.random.default_rng(SEED)

    counts = {c: {n: Counter() for n in nodes} for c in df.index}
    params_fixed_w = []

    for _ in range(N_SIM):
        q_s = {lf: rng.uniform(3, 5) for lf in leaves}
        p_s = {lf: rng.uniform(5, 7) for lf in leaves}
        v_s = {lf: rng.uniform(30, 50) for lf in leaves}
        b1_s = {lf: rng.uniform(35, 45) for lf in leaves}
        b2_s = {lf: rng.uniform(55, 65) for lf in leaves}
        b3_s = {lf: rng.uniform(75, 85) for lf in leaves}
        model.q, model.p, model.v = q_s, p_s, v_s
        model.profiles = {"b1": b1_s, "b2": b2_s, "b3": b3_s}
        params_fixed_w.append({"q": q_s, "p": p_s, "v": v_s, "b1": b1_s, "b2": b2_s, "b3": b3_s})

        for name, alt in alt_dicts.items():
            assigned = model.assign_all_nodes(alt, procedure="pessimistic")
            for n in nodes:
                counts[name][n][assigned[n]] += 1

    params_fixed_w_df = params_to_df(params_fixed_w)
    print(f"Simulation finished: {N_SIM} iterations.")
    print(f"Parameters stored in 'params_fixed_w' (list of dicts) and "
          f"'params_fixed_w_df' {params_fixed_w_df.shape}.")

    acceptability = build_acceptability(counts, df, nodes, categories)
    print(acceptability)
    print(class_acceptability_smaa(acceptability, categories, "Freedom_and_Prosperity").round(3))
    plot_acceptability_smaa(acceptability, categories, colors, "Freedom_and_Prosperity")
    plot_country_pies_smaa(acceptability, nodes, categories, colors, "Brazil")

    
    
    
    # SMAA-TRI — varying weights (Dirichlet from the root)
    print("\n" + "=" * 70)
    print("SMAA-TRI — varying weights")
    print("=" * 70)
    rng = np.random.default_rng(SEED)

    counts_w = {c: {n: Counter() for n in nodes} for c in df.index}
    params_varying_w = []

    for _ in range(N_SIM):
        w = sample_weights(Root, 1.0, rng)
        lam = compute_lambdas(Root, w, lambda_frac)
        q_s = {lf: rng.uniform(3, 5) for lf in leaves}
        p_s = {lf: rng.uniform(5, 7) for lf in leaves}
        v_s = {lf: rng.uniform(30, 50) for lf in leaves}
        b1_s = {lf: rng.uniform(35, 45) for lf in leaves}
        b2_s = {lf: rng.uniform(55, 65) for lf in leaves}
        b3_s = {lf: rng.uniform(75, 85) for lf in leaves}
        model.q, model.p, model.v = q_s, p_s, v_s
        model.weights, model.lambdas = w, lam
        model.profiles = {"b1": b1_s, "b2": b2_s, "b3": b3_s}

        params_varying_w.append(
            {"weights": w, "lambdas": lam,
             "q": q_s, "p": p_s, "v": v_s, "b1": b1_s, "b2": b2_s, "b3": b3_s}
        )

        for name, alt in alt_dicts.items():
            assigned = model.assign_all_nodes(alt, procedure="pessimistic")
            for n in nodes:
                counts_w[name][n][assigned[n]] += 1

    # restore the base-scenario parameters in the model
    model.weights, model.lambdas = weights, lambdas
    model.q, model.p, model.v, model.profiles = q, p, v, profiles

    params_varying_w_df = params_to_df(params_varying_w)
    print(f"Simulation finished: {N_SIM} iterations.")
    print(f"params_varying_w_df {params_varying_w_df.shape}.")

    acceptability_w = build_acceptability(counts_w, df, nodes, categories)
    print(acceptability_w)
    print(class_acceptability_smaa(acceptability_w, categories, "Freedom_and_Prosperity").round(3))
    plot_acceptability_smaa(acceptability_w, categories, colors, "Freedom_and_Prosperity")
    plot_country_pies_smaa(acceptability_w, nodes, categories, colors, "Brazil")

    
    
    
    
    # SMAA-TRI — Freedom/Prosperity split pinned at 0.5/0.5, children vary
    print("\n" + "=" * 70)
    print("SMAA-TRI — fixed Freedom/Prosperity split (0.5 / 0.5), varying children weights")
    print("=" * 70)
    rng = np.random.default_rng(SEED)

    counts_fp = {c: {n: Counter() for n in nodes} for c in df.index}
    params_fixed_fp = []

    for _ in range(N_SIM):
        # pin the top-level split: Freedom = 0.5, Prosperity = 0.5,
        # vary everything below via Dirichlet (full cascade)
        w = {}
        sample_weights(Freedom, 0.5, rng, w)
        sample_weights(Prosperity, 0.5, rng, w)
        lam = compute_lambdas(Root, w, lambda_frac)
        q_s = {lf: rng.uniform(3, 5) for lf in leaves}
        p_s = {lf: rng.uniform(5, 7) for lf in leaves}
        v_s = {lf: rng.uniform(30, 50) for lf in leaves}
        b1_s = {lf: rng.uniform(35, 45) for lf in leaves}
        b2_s = {lf: rng.uniform(55, 65) for lf in leaves}
        b3_s = {lf: rng.uniform(75, 85) for lf in leaves}
        model.q, model.p, model.v = q_s, p_s, v_s
        model.weights, model.lambdas = w, lam
        model.profiles = {"b1": b1_s, "b2": b2_s, "b3": b3_s}

        params_fixed_fp.append(
            {"weights": w, "lambdas": lam,
             "q": q_s, "p": p_s, "v": v_s, "b1": b1_s, "b2": b2_s, "b3": b3_s}
        )

        for name, alt in alt_dicts.items():
            assigned = model.assign_all_nodes(alt, procedure="pessimistic")
            for n in nodes:
                counts_fp[name][n][assigned[n]] += 1

    # restore the base-scenario parameters in the model
    model.weights, model.lambdas = weights, lambdas
    model.q, model.p, model.v, model.profiles = q, p, v, profiles

    params_fixed_fp_df = params_to_df(params_fixed_fp)
    print(f"Simulation finished: {N_SIM} iterations.")
    print(f"params_fixed_fp_df {params_fixed_fp_df.shape}.")

    acceptability_fp = build_acceptability(counts_fp, df, nodes, categories)
    print(acceptability_fp)
    print(class_acceptability_smaa(acceptability_fp, categories, "Freedom_and_Prosperity").round(3))
    plot_acceptability_smaa(acceptability_fp, categories, colors, "Freedom_and_Prosperity")
    plot_country_pies_smaa(acceptability_fp, nodes, categories, colors, "Brazil")


if __name__ == "__main__":
    main()
