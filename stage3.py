# -*- coding: utf-8 -*-
"""
Stage 3: Arrange ads inside each break (positions) + Final playlist

Reads:
- output/stage2_schedule.csv   : break_id, ad_id, deal_id, length_sec, target_demo, advertiser, brand, category, status
- data/deals_stage2.csv        : ad metadata (+ is_A_pos, is_Z_pos, piggyback_with, sandwich_with)
- data/ratings_stage2.csv      : break_id, demo_id, rating
- output/stage1_weights.csv    : deal_id, W_d
- data/breaks_stage2.csv       : break_id, length_sec, start_minute_F_b, hour

Outputs:
- output/stage3_positions.csv  : break_id, ad_id, position|BIN, gain_term, bin_penalty_term
- output/final_playlist.csv    : break_id, ad_id, position, start_time, end_time, length_sec, advertiser, brand
"""

from pathlib import Path
import pandas as pd
import pyomo.environ as pyo


# ---------- utils ----------
def _read_csv_first_exists(candidates):
    for p in map(Path, candidates):
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("None of these files exist:\n" + "\n".join(map(str, candidates)))


def get_solver(prefer="glpk"):
    for name in [prefer, "cbc", "glpk"]:
        try:
            opt = pyo.SolverFactory(name)
            if opt is not None and opt.available():
                return opt, name
        except Exception:
            pass
    raise RuntimeError("No MILP solver found. Please install GLPK or CBC via conda-forge.")


def hhmmss_from_sec(sec: int) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- 1) Load data ----------
def load_inputs(path_data: Path, path_out: Path):
    s2 = _read_csv_first_exists([path_out / "stage2_schedule.csv", "stage2_schedule.csv"])
    ds2 = _read_csv_first_exists([path_data / "deals_stage2.csv", "deals_stage2.csv"])
    r2  = _read_csv_first_exists([path_data / "ratings_stage2.csv", "ratings_stage2.csv"])
    w1  = _read_csv_first_exists([path_out / "stage1_weights.csv", "stage1_weights.csv"])
    brk = _read_csv_first_exists([path_data / "breaks_stage2.csv", "breaks_stage2.csv"])

    # Keep only scheduled rows
    s2 = s2[s2["status"].str.lower() == "scheduled"].copy()

    # Join specials
    ads = s2.merge(
        ds2,
        on=["ad_id","deal_id","length_sec","target_demo","advertiser","brand","category"],
        how="left"
    )

    # Join weights
    ads = ads.merge(w1[["deal_id","W_d"]], on="deal_id", how="left")

    # Ratings r_{b,q}
    r2 = r2.rename(columns={"demo_id": "target_demo", "rating": "rating_bq"})
    ads = ads.merge(r2, on=["break_id","target_demo"], how="left")

    # Checks
    missing = ads[ads["rating_bq"].isna()]
    if len(missing):
        raise ValueError(
            "Missing ratings for some (break_id, target_demo). "
            f"Examples:\n{missing[['break_id','ad_id','target_demo']].head()}"
        )
    if ads["W_d"].isna().any():
        raise ValueError("Some ads lack W_d (deal weight); check stage1_weights.csv.")

    # bin penalty rating = rating on current assigned break
    ads["rating_bin"] = ads["rating_bq"]

    # Flags normalize
    for col in ["is_A_pos", "is_Z_pos"]:
        if col in ads.columns:
            ads[col] = ads[col].fillna(False).astype(bool)
        else:
            ads[col] = False

    for col in ["piggyback_with", "sandwich_with"]:
        if col not in ads.columns:
            ads[col] = None

    return ads, brk


# ---------- 2) Solve Stage 3 for a single break ----------
def solve_stage3_for_break(df_b: pd.DataFrame, solver_name: str = "glpk", time_limit_sec: int = 60):
    ads = df_b["ad_id"].tolist()
    N = len(ads)
    positions = list(range(1, N+1))

    H = {r.ad_id: float(r.length_sec) for r in df_b.itertuples()}
    W = {r.ad_id: float(r.W_d) for r in df_b.itertuples()}
    r_gain = {r.ad_id: float(r.rating_bq) for r in df_b.itertuples()}
    r_bin  = {r.ad_id: float(r.rating_bin) for r in df_b.itertuples()}
    is_A = {r.ad_id: bool(r.is_A_pos) for r in df_b.itertuples()}
    is_Z = {r.ad_id: bool(r.is_Z_pos) for r in df_b.itertuples()}

    pig_pairs, snd_pairs = set(), set()
    for r in df_b.itertuples():
        if isinstance(r.piggyback_with, str) and r.piggyback_with in ads and r.piggyback_with != r.ad_id:
            a, b = sorted([r.ad_id, r.piggyback_with]); pig_pairs.add((a, b))
        if isinstance(r.sandwich_with, str) and r.sandwich_with in ads and r.sandwich_with != r.ad_id:
            a, b = sorted([r.ad_id, r.sandwich_with]); snd_pairs.add((a, b))

    m = pyo.ConcreteModel(name=f"Stage3_{df_b.break_id.iloc[0]}")
    m.ADS = pyo.Set(initialize=ads)
    m.POS = pyo.Set(initialize=positions, ordered=True)

    m.x = pyo.Var(m.ADS, m.POS, domain=pyo.Binary)
    m.y = pyo.Var(m.ADS, domain=pyo.Binary)
    m.p = pyo.Var(m.ADS, domain=pyo.Integers, bounds=(1, N))

    def obj_rule(mm):
        gain = sum(W[i]*(H[i]/30.0)*r_gain[i]*mm.x[i,l] for i in mm.ADS for l in mm.POS)
        pen  = sum(W[i]*(H[i]/30.0)*r_bin[i] *mm.y[i]   for i in mm.ADS)
        return gain - pen
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

    def one_place_or_bin(mm, i): return sum(mm.x[i,l] for l in mm.POS) + mm.y[i] == 1
    m.one_place_or_bin = pyo.Constraint(m.ADS, rule=one_place_or_bin)

    def one_ad_per_pos(mm, l): return sum(mm.x[i,l] for i in mm.ADS) <= 1
    m.one_ad_per_pos = pyo.Constraint(m.POS, rule=one_ad_per_pos)

    def link_pos(mm, i): return mm.p[i] == sum(l*mm.x[i,l] for l in mm.POS)
    m.link_pos = pyo.Constraint(m.ADS, rule=link_pos)

    A_ads = [i for i in ads if is_A[i]]
    if A_ads:
        def a_pos_rule(mm, i): return mm.x[i,1] == 1
        m.a_pos_con = pyo.Constraint(A_ads, rule=a_pos_rule)

    Z_ads = [i for i in ads if is_Z[i]]
    if Z_ads:
        def z_pos_rule(mm, i): return mm.x[i,N] == 1
        m.z_pos_con = pyo.Constraint(Z_ads, rule=z_pos_rule)

    M = max(10, N)
    if pig_pairs:
        m.PIG = pyo.Set(initialize=list(pig_pairs), dimen=2)
        m.s = pyo.Var(m.PIG, domain=pyo.Binary)
        m.pig_up  = pyo.Constraint(m.PIG, rule=lambda mm,i,j: mm.p[i]-mm.p[j] <= 1)
        m.pig_dn  = pyo.Constraint(m.PIG, rule=lambda mm,i,j: mm.p[j]-mm.p[i] <= 1)
        m.pig_d1  = pyo.Constraint(m.PIG, rule=lambda mm,i,j: mm.p[i]-mm.p[j] >= 1 - M*(1-mm.s[i,j]))
        m.pig_d2  = pyo.Constraint(m.PIG, rule=lambda mm,i,j: mm.p[j]-mm.p[i] >= 1 - M*(   mm.s[i,j]))

    if snd_pairs:
        m.SND = pyo.Set(initialize=list(snd_pairs), dimen=2)
        m.t = pyo.Var(m.SND, domain=pyo.Binary)
        m.snd_up = pyo.Constraint(m.SND, rule=lambda mm,i,j: mm.p[i]-mm.p[j] <= 2)
        m.snd_dn = pyo.Constraint(m.SND, rule=lambda mm,i,j: mm.p[j]-mm.p[i] <= 2)
        m.snd_d1 = pyo.Constraint(m.SND, rule=lambda mm,i,j: mm.p[i]-mm.p[j] >= 2 - M*(1-mm.t[i,j]))
        m.snd_d2 = pyo.Constraint(m.SND, rule=lambda mm,i,j: mm.p[j]-mm.p[i] >= 2 - M*(   mm.t[i,j]))

    opt, used = get_solver(prefer=solver_name)
    try: opt.options["tmlim"] = time_limit_sec
    except Exception: pass
    results = opt.solve(m, tee=False)

    rows = []
    for i in ads:
        if pyo.value(m.y[i]) > 0.5:
            rows.append({"ad_id": i, "decision": "BIN", "gain_term": 0.0,
                         "bin_penalty_term": round(float(W[i]*(H[i]/30.0)*r_bin[i]), 6)})
        else:
            pos = next(l for l in positions if pyo.value(m.x[i,l]) > 0.5)
            rows.append({"ad_id": i, "decision": f"pos={int(pos)}",
                         "gain_term": round(float(W[i]*(H[i]/30.0)*r_gain[i]), 6),
                         "bin_penalty_term": 0.0})
    return rows, results, used


# ---------- 3) Run all breaks & save ----------
def run_stage3(path_data: Path, path_out: Path,
               solver_name: str = "glpk", time_limit_sec: int = 60,
               outfile_positions: str = "stage3_positions.csv",
               outfile_playlist: str = "final_playlist.csv"):

    ads_all, breaks_df = load_inputs(path_data, path_out)

    # per-break solve
    outputs = []
    solver_used = None
    for b, df_b in ads_all.groupby("break_id"):
        rows, _, used = solve_stage3_for_break(df_b, solver_name, time_limit_sec)
        solver_used = used
        for r in rows:
            if r["decision"] == "BIN":
                outputs.append({
                    "break_id": b, "ad_id": r["ad_id"], "position": "BIN",
                    "gain_term": r["gain_term"], "bin_penalty_term": r["bin_penalty_term"]
                })
            else:
                pos = int(r["decision"].split("=")[1])
                outputs.append({
                    "break_id": b, "ad_id": r["ad_id"], "position": pos,
                    "gain_term": r["gain_term"], "bin_penalty_term": r["bin_penalty_term"]
                })

    pos_df = pd.DataFrame(outputs)

    # 排序（BIN 最后）
    pos_df["pos_order"] = pd.to_numeric(pos_df["position"], errors="coerce").fillna(10**9)
    pos_df = pos_df.sort_values(by=["break_id", "pos_order", "ad_id"], kind="mergesort").drop(columns=["pos_order"])

    # 保存 stage3_positions.csv
    path_out.mkdir(parents=True, exist_ok=True)
    positions_path = path_out / outfile_positions
    pos_df.to_csv(positions_path, index=False, encoding="utf-8-sig")
    print(f"[Stage 3] Solver: {solver_used} | Saved positions -> {positions_path}")

    # ---------- Final playlist ----------
    # 只保留已排位（非 BIN）
    placed = pos_df[pos_df["position"] != "BIN"].copy()

    # 合并广告时长/元数据
    deals_df = _read_csv_first_exists([path_data/"deals_stage2.csv", "deals_stage2.csv"])
    placed = placed.merge(
        deals_df[["ad_id","length_sec","advertiser","brand"]],
        on="ad_id", how="left"
    )

    # 合并 break 起始时间（分钟 -> 秒）
    brk = breaks_df[["break_id","start_minute_F_b"]].copy()
    brk["break_start_sec"] = brk["start_minute_F_b"] * 60
    placed = placed.merge(brk[["break_id","break_start_sec"]], on="break_id", how="left")

    # 按 break / position 计算起止时间
    placed = placed.sort_values(["break_id","position"])
    placed["start_time_sec"] = placed.groupby("break_id").apply(
        lambda g: g["length_sec"].cumsum() - g["length_sec"]
    ).reset_index(level=0, drop=True) + placed["break_start_sec"]
    placed["end_time_sec"] = placed["start_time_sec"] + placed["length_sec"]

    placed["start_time"] = placed["start_time_sec"].apply(hhmmss_from_sec)
    placed["end_time"]   = placed["end_time_sec"].apply(hhmmss_from_sec)

    final_cols = ["break_id","ad_id","position","start_time","end_time","length_sec","advertiser","brand"]
    playlist = placed[final_cols].copy()

    playlist_path = path_out / outfile_playlist
    playlist.to_csv(playlist_path, index=False, encoding="utf-8-sig")
    print(f"[Stage 3] Saved broadcast-ready playlist -> {playlist_path}")

    # 简报
    bin_cnt = (pos_df["position"] == "BIN").sum()
    print(f"[Stage 3] Placed: {len(playlist)} rows | Binned: {bin_cnt}")

    return pos_df, playlist


if __name__ == "__main__":
    data_dir = Path("./data")
    out_dir  = Path("./output")
    run_stage3(
        path_data=data_dir,
        path_out=out_dir,
        solver_name="glpk",        # 可改 "cbc"
        time_limit_sec=60,
        outfile_positions="stage3_positions.csv",
        outfile_playlist="final_playlist.csv"
    )
