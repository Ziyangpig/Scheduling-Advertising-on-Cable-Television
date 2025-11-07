"""
Stage 1 (Weight Problem) 
- Reads:
    data/deals_stage1.csv
    data/breaks_stage1.csv
    data/ratings_stage1.csv
- Builds LP:
    min sum_d CPM_d * y_d
    s.t.
        sum_d H * x_{db} <= L_b        (break capacity)
        sum_b r_{q(d),b} * x_{db} + y_d >= I_d   (impression target)
        sum_b x_{db} <= J_d            (max number of ads per deal)
- Outputs:
    output/stage1_weights.csv : deal_id, W_d, y_d
    output/stage1_xdb.csv     : deal_id, break_id, x_db
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import csv
from pathlib import Path

from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    NonNegativeReals,
    Objective,
    Constraint,
    minimize,
    Suffix,
    value,
    SolverFactory,
)


# --------- Data classes for clean interface ---------


@dataclass
class Stage1Input:
    deals_csv: str
    breaks_csv: str
    ratings_csv: str
    avg_ad_length: float = 30.0  # H, average ad length in seconds
    solver_name: str = "glpk"    # you can change to "cbc", "gurobi", etc.


@dataclass
class Stage1Output:
    weights: Dict[str, float]                # W_d, duals of impression constraints
    shortfall: Dict[str, float]              # y_d
    x_db: Dict[Tuple[str, str], float]       # x_{d,b}


# --------- CSV loaders ---------


def load_deals(path: str):
    """Load deals and basic parameters used in Stage 1."""
    deals = []
    target_demo = {}
    I_d = {}
    J_d = {}
    CPM_d = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = row["deal_id"]
            deals.append(d)
            target_demo[d] = row["target_demo"]
            I_d[d] = float(row["I_d"])
            J_d[d] = float(row["J_d"])
            CPM_d[d] = float(row["CPM_d"])

    return deals, target_demo, I_d, J_d, CPM_d


def load_breaks(path: str):
    """Load breaks and their lengths."""
    breaks = []
    L_b = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = row["break_id"]
            breaks.append(b)
            L_b[b] = float(row["length_sec"])

    return breaks, L_b


def load_ratings(path: str):
    """Load ratings: (demo_id, break_id) -> rating."""
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b = row["break_id"]
            q = row["demo_id"]
            ratings[(q, b)] = float(row["rating"])
    return ratings


# --------- Build Stage 1 data structure ---------


def build_stage1_data(inp: Stage1Input):
    """
    Build all data needed by the Stage 1 model from CSV files.
    """
    D, target_demo, I_d, J_d, CPM_d = load_deals(inp.deals_csv)
    B, L_b = load_breaks(inp.breaks_csv)
    ratings = load_ratings(inp.ratings_csv)

    # BD(d): allowed breaks for each deal = all breaks where rating for q(d) exists
    BD = {d: [] for d in D}
    # r_db: ratings for each (deal, break) pair
    r_db = {}

    for d in D:
        q = target_demo[d]
        for b in B:
            key = (q, b)
            if key in ratings:
                BD[d].append(b)
                r_db[(d, b)] = ratings[key]
        # In a real system, BD(d) can be further restricted by contract rules.

    data = {
        "D": D,
        "B": B,
        "BD": BD,
        "L_b": L_b,
        "I_d": I_d,
        "J_d": J_d,
        "CPM_d": CPM_d,
        "r_db": r_db,
        "H": inp.avg_ad_length,
    }

    return data


# --------- Build Pyomo model ---------


def build_stage1_model(data) -> ConcreteModel:
    D = data["D"]
    B = data["B"]
    BD = data["BD"]
    L_b = data["L_b"]
    I_d = data["I_d"]
    J_d = data["J_d"]
    CPM_d = data["CPM_d"]
    r_db = data["r_db"]
    H = data["H"]

    m = ConcreteModel()

    # Sets
    m.D = Set(initialize=D)
    m.B = Set(initialize=B)
    m.DB = Set(within=m.D * m.B,
               initialize=[(d, b) for d in D for b in BD[d]])

    # Variables
    m.x = Var(m.DB, domain=NonNegativeReals)   # x_{d,b}
    m.y = Var(m.D, domain=NonNegativeReals)    # y_d

    # Objective: min sum_d CPM_d * y_d
    def obj_rule(m):
        return sum(CPM_d[d] * m.y[d] for d in m.D)

    m.obj = Objective(rule=obj_rule, sense=minimize)

    # Break capacity: sum_d H * x_{d,b} <= L_b
    def break_capacity_rule(m, b):
        return sum(H * m.x[d, b] for d in m.D if (d, b) in m.DB) <= L_b[b]

    m.break_capacity = Constraint(m.B, rule=break_capacity_rule)

    # Impression target: sum_b r_{q(d),b} * x_{d,b} + y_d >= I_d
    def impression_rule(m, d):
        return sum(r_db[(d, b)] * m.x[d, b] for b in BD[d]) + m.y[d] >= I_d[d]

    m.impression_target = Constraint(m.D, rule=impression_rule)

    # Max number of ads per deal: sum_b x_{d,b} <= J_d
    def max_ads_rule(m, d):
        return sum(m.x[d, b] for b in BD[d]) <= J_d[d]

    m.max_ads = Constraint(m.D, rule=max_ads_rule)

    # Suffix for dual prices (shadow prices)
    m.dual = Suffix(direction=Suffix.IMPORT)

    return m


# --------- Solve and extract results ---------


def solve_stage1(model: ConcreteModel, solver_name: str) -> Stage1Output:
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=False)

    # Extract x_{d,b}
    x_db: Dict[Tuple[str, str], float] = {}
    for (d, b) in model.DB:
        v = value(model.x[d, b])
        if v is not None and v > 1e-6:
            x_db[(d, b)] = float(v)

    # Extract y_d
    shortfall: Dict[str, float] = {}
    for d in model.D:
        v = value(model.y[d])
        shortfall[d] = float(v if v is not None else 0.0)

    # Extract W_d = dual of impression_target[d]
    weights: Dict[str, float] = {}
    for d in model.D:
        dual_val = model.dual.get(model.impression_target[d], 0.0)
        weights[d] = float(dual_val)

    return Stage1Output(weights=weights, shortfall=shortfall, x_db=x_db)


def run_stage1(inp: Stage1Input) -> Stage1Output:
    """High-level API: from CSV -> Stage1Output."""
    data = build_stage1_data(inp)
    model = build_stage1_model(data)
    return solve_stage1(model, inp.solver_name)


def save_stage1_results(output: Stage1Output,
                        weights_path: str,
                        xdb_path: str) -> None:
    """Save Stage 1 results to CSV files for other team members."""
    weights_file = Path(weights_path)
    xdb_file = Path(xdb_path)
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    xdb_file.parent.mkdir(parents=True, exist_ok=True)

    # Save W_d and y_d
    with open(weights_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["deal_id", "W_d", "y_d"])
        for d in sorted(output.weights.keys()):
            writer.writerow([
                d,
                f"{output.weights[d]:.6f}",
                f"{output.shortfall.get(d, 0.0):.6f}",
            ])

    # Save x_{d,b}
    with open(xdb_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["deal_id", "break_id", "x_db"])
        for (d, b) in sorted(output.x_db.keys()):
            writer.writerow([
                d,
                b,
                f"{output.x_db[(d, b)]:.6f}",
            ])


# --------- Command-line entry point ---------


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "output"

    inp = Stage1Input(
        deals_csv=str(data_dir / "deals_stage1.csv"),
        breaks_csv=str(data_dir / "breaks_stage1.csv"),
        ratings_csv=str(data_dir / "ratings_stage1.csv"),
        avg_ad_length=30.0,
        solver_name="glpk",  
    )

    result = run_stage1(inp)
    save_stage1_results(
        result,
        weights_path=str(out_dir / "stage1_weights.csv"),
        xdb_path=str(out_dir / "stage1_xdb.csv"),
    )

    print("Stage 1 finished. Results saved to", out_dir)
