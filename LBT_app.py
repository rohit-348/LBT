import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Cryo LBT Benchmarking Studio",
    page_icon="⚡",
    layout="wide",
)

# ==============================
# CORE LBT PROCESSING FUNCTIONS
# ==============================

def load_sheets(uploaded_file, sheet_t, sheet_c, sheet_l):
    T = pd.read_excel(uploaded_file, sheet_name=sheet_t)
    C = pd.read_excel(uploaded_file, sheet_name=sheet_c)
    L = pd.read_excel(uploaded_file, sheet_name=sheet_l)

    T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
    C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")
    return T, C, L


def parse_lbt_to_state(L_df, available_tags):
    """Converts the raw lbt_input DataFrame into clean, structured UI state."""
    tags = [str(t).strip() for t in L_df["Match_tags"].dropna().unique().tolist()]
    if not tags and available_tags:
        tags = [available_tags[0]]

    # Decimal rounding per tag
    decimals = {}
    if "Match_Tag_Decimal" in L_df.columns:
        valid_dec = L_df[["Match_tags", "Match_Tag_Decimal"]].dropna()
        for _, r in valid_dec.iterrows():
            decimals[str(r["Match_tags"]).strip()] = int(r["Match_Tag_Decimal"])
    for t in tags:
        decimals.setdefault(t, 2)

    # Iterations dictionary: {1: {tag: {"min": x, "max": y}}, 2: ...}
    iterations = {}
    iter_nums = sorted(pd.to_numeric(L_df["Iteration No"], errors="coerce").dropna().astype(int).unique().tolist())
    if not iter_nums:
        iter_nums = [1]

    for it in iter_nums:
        iterations[it] = {}
        sub = L_df[pd.to_numeric(L_df["Iteration No"], errors="coerce") == it]
        for t in tags:
            row_match = sub[sub["Match_tags"].astype(str).str.strip() == t]
            if not row_match.empty:
                min_v = pd.to_numeric(row_match.iloc[0]["Tolerance_minimum"], errors="coerce")
                max_v = pd.to_numeric(row_match.iloc[0]["Tolerance_maximum"], errors="coerce")
                iterations[it][t] = {
                    "min": float(min_v) if pd.notnull(min_v) else 0.0,
                    "max": float(max_v) if pd.notnull(max_v) else 0.0,
                }
            else:
                iterations[it][t] = {"min": 0.0, "max": 0.0}

    # Performance tag & direction
    perf_tag = available_tags[0] if available_tags else ""
    direction = "minimize"
    if "performance_tag" in L_df.columns and "direction" in L_df.columns:
        valid_perf = L_df[["performance_tag", "direction"]].dropna()
        if not valid_perf.empty:
            perf_tag = str(valid_perf.iloc[0]["performance_tag"]).strip()
            direction = str(valid_perf.iloc[0]["direction"]).strip().lower()

    return {
        "tags": tags,
        "decimals": decimals,
        "iterations": iterations,
        "perf_tag": perf_tag,
        "direction": direction if direction in ["minimize", "maximize"] else "minimize",
    }


def build_lbt_df_from_state(cfg):
    """Reconstructs a clean lbt_input DataFrame from the structured UI state."""
    rows = []
    sorted_iters = sorted(cfg["iterations"].keys())
    for it in sorted_iters:
        for t in cfg["tags"]:
            tol = cfg["iterations"][it].get(t, {"min": 0.0, "max": 0.0})
            rows.append({
                "Match_tags": t,
                "Match_Tag_Decimal": int(cfg["decimals"].get(t, 2)),
                "Iteration No": int(it),
                "Tolerance_minimum": float(tol["min"]),
                "Tolerance_maximum": float(tol["max"]),
                "performance_tag": None,
                "direction": None,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["performance_tag"] = df["performance_tag"].astype("object")
        df["direction"] = df["direction"].astype("object")
        df.loc[0, "performance_tag"] = str(cfg["perf_tag"])
        df.loc[0, "direction"] = str(cfg["direction"])
    return df


def add_feed_stability(df, feed_col, min_val, max_val):
    if feed_col not in df.columns:
        st.warning(f"⚠️ Feed column '{feed_col}' not found in clean data. Skipping stability check.")
        return df

    stable_mask = (df[feed_col] >= min_val) & (df[feed_col] <= max_val)
    df["Feed_Stable"] = stable_mask.map({True: "Stable", False: "Unstable"})
    return df


def build_result(row_t, matched_row, T, C, L, match_tags, decimal_dict):
    result = {}
    for col in T.columns:
        val_actual = row_t[col]
        val_benchmark = matched_row.get(col, np.nan)
        if col in decimal_dict:
            decimals = int(decimal_dict[col])
            val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
            val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark
        result[f"{col}_Actual"] = val_actual
        result[f"{col}_benchmark"] = val_benchmark

    for col in C.columns:
        if col not in T.columns:
            val_benchmark = matched_row.get(col, np.nan)
            if col in decimal_dict:
                decimals = int(decimal_dict[col])
                val_benchmark = round(val_benchmark, decimals) if pd.notnull(val_benchmark) else val_benchmark
            result[f"{col}_benchmark"] = val_benchmark

    matched_iterations = []
    for tag in match_tags:
        delta_col = f"{tag}_delta"
        iter_col = f"matched_iteration_{tag}"
        actual_val = row_t.get(tag, np.nan)
        benchmark_val = matched_row.get(tag, np.nan)
        if pd.notnull(actual_val) and pd.notnull(benchmark_val):
            decimals = int(decimal_dict.get(tag, 4))
            actual_val = round(actual_val, decimals)
            benchmark_val = round(benchmark_val, decimals)
            delta = round(benchmark_val - actual_val, decimals)
            result[delta_col] = delta
            found_iter = None
            for _, tol_row in L[L["Match_tags"] == tag].iterrows():
                it_no = tol_row["Iteration No"]
                min_tol = tol_row["Tolerance_minimum"]
                max_tol = tol_row["Tolerance_maximum"]
                if pd.notnull(min_tol) and pd.notnull(max_tol) and (-min_tol <= delta <= max_tol):
                    found_iter = it_no
                    break
            result[iter_col] = found_iter
            if found_iter is not None:
                matched_iterations.append(found_iter)
        else:
            result[delta_col] = np.nan
            result[iter_col] = None

    result["iteration_matched"] = max(matched_iterations) if matched_iterations else None
    return result


def build_no_match_result(row_t, T, C, match_tags, decimal_dict):
    result = {}
    for col in T.columns:
        val_actual = row_t[col]
        if col in decimal_dict:
            decimals = int(decimal_dict[col])
            val_actual = round(val_actual, decimals) if pd.notnull(val_actual) else val_actual
        result[f"{col}_Actual"] = val_actual
        result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

    for col in C.columns:
        if col not in T.columns:
            result[f"{col}_benchmark"] = "BENCHMARK NOT FOUND"

    for tag in match_tags:
        result[f"{tag}_delta"] = "BENCHMARK NOT FOUND"
        result[f"matched_iteration_{tag}"] = "BENCHMARK NOT FOUND"

    result["iteration_matched"] = "BENCHMARK NOT FOUND"
    return result


def match_records(T, C, L, max_benchmarks, decimal_dict):
    results = []
    match_tags = L["Match_tags"].dropna().unique().tolist()
    iter_series = pd.to_numeric(L["Iteration No"], errors="coerce").dropna()
    if iter_series.empty:
        raise ValueError("No valid iterations configured.")
    max_iterations = int(iter_series.max())

    for _, row_t in T.iterrows():
        match_found = False
        for iteration in range(1, max_iterations + 1):
            current_tol = L[pd.to_numeric(L["Iteration No"], errors="coerce") == iteration]
            filtered_C = C.copy()
            for _, tol_row in current_tol.iterrows():
                col = tol_row["Match_tags"]
                min_tol = tol_row["Tolerance_minimum"]
                max_tol = tol_row["Tolerance_maximum"]
                if (
                    pd.isnull(col)
                    or col not in T.columns
                    or col not in C.columns
                    or col == "Timestamp"
                    or pd.isnull(min_tol)
                    or pd.isnull(max_tol)
                ):
                    continue
                val_t = row_t[col]
                if pd.isnull(val_t):
                    continue
                if col in decimal_dict:
                    decimals = int(decimal_dict[col])
                    val_t = round(val_t, decimals)
                    filtered_C[col] = filtered_C[col].round(decimals)
                filtered_C = filtered_C[
                    (filtered_C[col] >= val_t - min_tol) & (filtered_C[col] <= val_t + max_tol)
                ]
            if not filtered_C.empty:
                match_found = True
                for _, matched_row in filtered_C.head(max_benchmarks).iterrows():
                    results.append(build_result(row_t, matched_row, T, C, L, match_tags, decimal_dict))
                break
        if not match_found:
            results.append(build_no_match_result(row_t, T, C, match_tags, decimal_dict))
    return results


def filter_optimal_rows(df, L):
    perf_config = L[["performance_tag", "direction"]].dropna().drop_duplicates()
    if perf_config.empty:
        return df

    perf_tag = str(perf_config.iloc[0]["performance_tag"]).strip()
    direction = str(perf_config.iloc[0]["direction"]).strip().lower()
    perf_col = f"{perf_tag}_benchmark"
    if perf_col not in df.columns:
        st.warning(f"⚠️ Performance column '{perf_col}' not found - skipping optimal filtering.")
        return df

    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")
    matched_df = df.dropna(subset=[perf_col])
    unmatched_df = df[df[perf_col].isna()]

    if direction == "minimize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmin()]
    elif direction == "maximize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmax()]
    else:
        raise ValueError(f"Unknown direction '{direction}' in performance config.")

    unmatched_latest = unmatched_df.sort_values("Timestamp_Actual").drop_duplicates(
        subset=["Timestamp_Actual"], keep="last"
    )

    return pd.concat(
        [optimal_df, unmatched_latest[~unmatched_latest["Timestamp_Actual"].isin(optimal_df["Timestamp_Actual"])]],
        ignore_index=True,
    ).sort_values("Timestamp_Actual").reset_index(drop=True)


def process_results(results, T, C, L):
    df = pd.DataFrame(results)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    for col in [c for c in df.columns if c.endswith("Timestamp_benchmark")]:
        df[col] = df[col].fillna("BENCHMARK NOT FOUND")

    actual_bench_cols = []
    for col in T.columns:
        if f"{col}_Actual" in df.columns:
            actual_bench_cols.append(f"{col}_Actual")
        if f"{col}_benchmark" in df.columns:
            actual_bench_cols.append(f"{col}_benchmark")

    for col in C.columns:
        if col not in T.columns and f"{col}_benchmark" in df.columns:
            actual_bench_cols.append(f"{col}_benchmark")

    delta_cols = sorted([c for c in df.columns if c.endswith("_delta")])
    iter_cols = sorted([c for c in df.columns if c.startswith("matched_iteration_")])
    final_cols = actual_bench_cols + delta_cols + iter_cols
    if "iteration_matched" in df.columns:
        final_cols.append("iteration_matched")
    remaining = [c for c in df.columns if c not in final_cols]
    df = df[final_cols + remaining]
    return filter_optimal_rows(df, L)


def generate_plot_and_excel_bytes(df, used_L_df, plot_sheet_name="LinePlot"):
    plot_df = df.copy()
    plot_df["Timestamp_Actual"] = pd.to_datetime(plot_df["Timestamp_Actual"], errors="coerce")
    iter_cols = [col for col in plot_df.columns if col.startswith("matched_iteration_")]

    fig = None
    plot_buf = None

    if iter_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in iter_cols:
            numeric_series = pd.to_numeric(plot_df[col], errors="coerce")
            ax.plot(
                plot_df["Timestamp_Actual"],
                numeric_series,
                marker="o",
                markersize=4,
                label=col.replace("matched_iteration_", ""),
            )

        ax.set_ylabel("Iteration Number")
        ax.set_title("Match Tag Iterations")
        x_labels = plot_df["Timestamp_Actual"].dt.strftime("%d %b %y")
        ax.set_xticks(plot_df["Timestamp_Actual"])
        ax.set_xticklabels(x_labels, rotation=60)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        plt.tight_layout()

        plot_buf = io.BytesIO()
        fig.savefig(plot_buf, format="png", dpi=150)
        plot_buf.seek(0)

    # Sheet 1: Edited LBT Input Configuration
    # Sheet 2: Benchmark Results Data
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        used_L_df.to_excel(writer, index=False, sheet_name="lbt_input")
        df.to_excel(writer, index=False, sheet_name="Results")

    # Sheet 3: Embedded Line Plot
    if plot_buf is not None:
        excel_buf.seek(0)
        wb = load_workbook(excel_buf)
        ws = wb[plot_sheet_name] if plot_sheet_name in wb.sheetnames else wb.create_sheet(plot_sheet_name)
        img = Image(plot_buf)
        img.anchor = "A1"
        ws.add_image(img)

        final_excel_buf = io.BytesIO()
        wb.save(final_excel_buf)
        final_excel_buf.seek(0)
        return final_excel_buf, fig

    excel_buf.seek(0)
    return excel_buf, fig


# ==============================
# STREAMLIT UI LAYOUT
# ==============================

st.title("⚡ Look-Back Benchmarking (LBT) Studio")
st.markdown("Upload your Excel workbook to inspect and customize stability limits, KPI targets, and iteration tolerances.")

# --- Step 1: Sheet Names & File Uploader ---
with st.expander("📄 Sheet Name Mapping (Optional — edit only if your workbook uses custom tab names)", expanded=False):
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sheet_t = st.text_input("Test Data Sheet", value="test_data")
    with s_col2:
        sheet_c = st.text_input("Clean Data Sheet", value="clean_data")
    with s_col3:
        sheet_l = st.text_input("LBT Config Sheet", value="lbt_input")

uploaded_file = st.file_uploader("📥 Step 1: Upload Input Excel File (.xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_file_id") != file_id:
        try:
            T_raw, C_raw, L_raw = load_sheets(uploaded_file, sheet_t, sheet_c, sheet_l)
            available_tags = [c for c in C_raw.columns if c != "Timestamp"]
            parsed_cfg = parse_lbt_to_state(L_raw, available_tags)

            st.session_state["current_file_id"] = file_id
            st.session_state["T_raw"] = T_raw
            st.session_state["C_raw"] = C_raw
            st.session_state["L_raw"] = L_raw
            st.session_state["available_tags"] = available_tags
            st.session_state["cfg"] = parsed_cfg
            st.session_state.pop("final_df", None)
        except Exception as e:
            st.error(f"❌ Could not read sheets from Excel file: {e}")

# --- Step 2: Global Settings (Appears ONLY after file is uploaded) ---
if "cfg" in st.session_state:
    cfg = st.session_state["cfg"]
    available_tags = st.session_state["available_tags"]

    st.divider()
    top_bar_l, top_bar_r = st.columns([4, 1])
    with top_bar_l:
        st.header("⚙️ Step 2: Global Settings & Iteration Tolerances")
    with top_bar_r:
        if st.button("↺ Reset All to Excel Defaults", use_container_width=True):
            st.session_state["cfg"] = parse_lbt_to_state(st.session_state["L_raw"], available_tags)
            st.rerun()

    # --- Section A: Feed Stability & General Engine Parameters ---
    with st.container(border=True):
        st.subheader("1️⃣ Feed Stability & Matching Rules")
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            default_feed_idx = available_tags.index("feed_MMSCFD") if "feed_MMSCFD" in available_tags else 0
            feed_tag = st.selectbox(
                "Feed Stability Tag",
                options=available_tags if available_tags else ["feed_MMSCFD"],
                index=default_feed_idx,
                help="Column in clean_data used to filter stable historical operating periods.",
            )
        with c2:
            feed_min = st.number_input("Min Stable Feed", value=180.0, step=5.0)
        with c3:
            feed_max = st.number_input("Max Stable Feed", value=250.0, step=5.0)
        with c4:
            max_benchmarks = st.number_input("Max Benchmarks / Match", min_value=1, max_value=100, value=10, step=1)

    # --- Section B: KPI Optimization & Active Match Tags ---
    with st.container(border=True):
        st.subheader("2️⃣ Optimization Objective & Active Match Tags")
        k1, k2 = st.columns([2, 1])
        with k1:
            perf_idx = available_tags.index(cfg["perf_tag"]) if cfg["perf_tag"] in available_tags else 0
            cfg["perf_tag"] = st.selectbox(
                "Target Performance Tag (KPI)",
                options=available_tags,
                index=perf_idx,
                help="The benchmark parameter used to pick the single best historical row per timestamp.",
            )
        with k2:
            dir_opts = ["minimize", "maximize"]
            cfg["direction"] = st.selectbox(
                "Optimization Direction",
                options=dir_opts,
                index=dir_opts.index(cfg["direction"]),
            )

        selected_tags = st.multiselect(
            "Active Match Tags (Tags compared across iterations)",
            options=available_tags,
            default=[t for t in cfg["tags"] if t in available_tags],
        )
        if selected_tags != cfg["tags"]:
            cfg["tags"] = selected_tags
            for t in selected_tags:
                cfg["decimals"].setdefault(t, 2)
                for it in cfg["iterations"]:
                    cfg["iterations"][it].setdefault(t, {"min": 0.0, "max": 0.0})

    # --- Section C: User-Friendly Iteration Tolerance Editor ---
    with st.container(border=True):
        it_head_l, it_head_r1, it_head_r2 = st.columns([3, 1, 1])
        with it_head_l:
            st.subheader("3️⃣ Iteration Tolerance Windows")
            st.caption("Select an iteration tab below to adjust the allowable negative (`- Min`) and positive (`+ Max`) deviation for each tag.")
        with it_head_r1:
            if st.button("➕ Add Iteration", use_container_width=True):
                existing_iters = sorted(cfg["iterations"].keys())
                new_it = (existing_iters[-1] + 1) if existing_iters else 1
                prev_it = existing_iters[-1] if existing_iters else None
                cfg["iterations"][new_it] = {}
                for t in cfg["tags"]:
                    if prev_it is not None and t in cfg["iterations"][prev_it]:
                        cfg["iterations"][new_it][t] = {
                            "min": round(cfg["iterations"][prev_it][t]["min"] * 1.5, 4),
                            "max": round(cfg["iterations"][prev_it][t]["max"] * 1.5, 4),
                        }
                    else:
                        cfg["iterations"][new_it][t] = {"min": 0.0, "max": 0.0}
                st.rerun()
        with it_head_r2:
            if len(cfg["iterations"]) > 1:
                if st.button("🗑️ Remove Last Iteration", use_container_width=True):
                    last_it = max(cfg["iterations"].keys())
                    del cfg["iterations"][last_it]
                    st.rerun()

        sorted_iters = sorted(cfg["iterations"].keys())
        if not cfg["tags"]:
            st.warning("⚠️ Please select at least one Active Match Tag above.")
        else:
            iter_tabs = st.tabs([f"Iteration {it}" for it in sorted_iters])

            for idx, it in enumerate(sorted_iters):
                with iter_tabs[idx]:
                    h_tag, h_dec, h_min, h_max = st.columns([3, 1, 1.5, 1.5])
                    h_tag.markdown("**Process Tag**")
                    h_dec.markdown("**Decimals**")
                    h_min.markdown("**Min Tolerance (-)**")
                    h_max.markdown("**Max Tolerance (+)**")

                    for t in cfg["tags"]:
                        r_tag, r_dec, r_min, r_max = st.columns([3, 1, 1.5, 1.5])
                        with r_tag:
                            st.markdown(f"<div style='padding-top: 8px; font-weight: 500;'>{t}</div>", unsafe_allow_html=True)
                        with r_dec:
                            cfg["decimals"][t] = st.number_input(
                                f"dec_{it}_{t}",
                                min_value=0,
                                max_value=8,
                                value=int(cfg["decimals"].get(t, 2)),
                                step=1,
                                label_visibility="collapsed",
                            )
                        with r_min:
                            cfg["iterations"][it][t]["min"] = st.number_input(
                                f"min_{it}_{t}",
                                value=float(cfg["iterations"][it][t]["min"]),
                                format="%.4f",
                                label_visibility="collapsed",
                            )
                        with r_max:
                            cfg["iterations"][it][t]["max"] = st.number_input(
                                f"max_{it}_{t}",
                                value=float(cfg["iterations"][it][t]["max"]),
                                format="%.4f",
                                label_visibility="collapsed",
                            )

    # --- Run Benchmarking Button ---
    st.markdown("")
    if st.button("▶ Run Benchmarking Analysis", type="primary", use_container_width=True):
        if not cfg["tags"]:
            st.error("❌ Please select at least one match tag before running.")
        else:
            with st.spinner("Filtering stable periods and matching tolerance iterations..."):
                try:
                    T = st.session_state["T_raw"].copy()
                    C = st.session_state["C_raw"].copy()
                    L_active = build_lbt_df_from_state(cfg)
                    decimal_dict = {t: int(cfg["decimals"][t]) for t in cfg["tags"]}

                    total_clean_rows = len(C)

                    # 1. Stability Filter
                    C = add_feed_stability(C, feed_col=feed_tag, min_val=feed_min, max_val=feed_max)
                    if "Feed_Stable" in C.columns:
                        C = C[C["Feed_Stable"] == "Stable"].copy()
                    stable_clean_rows = len(C)

                    # 2. Match & Optimize
                    results = match_records(T, C, L_active, int(max_benchmarks), decimal_dict)
                    final_df = process_results(results, T, C, L_active)

                    # 3. Generate Plot & Output Excel (Sheet 1: lbt_input, Sheet 2: Results, Sheet 3: LinePlot)
                    excel_bytes, fig = generate_plot_and_excel_bytes(final_df, L_active)

                    st.session_state["final_df"] = final_df
                    st.session_state["used_L_df"] = L_active
                    st.session_state["excel_bytes"] = excel_bytes
                    st.session_state["fig"] = fig
                    st.session_state["metrics"] = {
                        "test_rows": len(T),
                        "stable_rows": stable_clean_rows,
                        "total_clean": total_clean_rows,
                        "matched_rows": int((final_df["iteration_matched"] != "BENCHMARK NOT FOUND").sum())
                        if "iteration_matched" in final_df.columns else 0,
                    }
                    st.success("✅ Benchmarking analysis completed!")

                except Exception as e:
                    st.error(f"❌ Error during execution: {e}")

# --- Step 3: Results Dashboard ---
if "final_df" in st.session_state:
    st.divider()
    st.header("📊 Step 3: Benchmarking Results")
    metrics = st.session_state["metrics"]
    final_df = st.session_state["final_df"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Records Evaluated", metrics["test_rows"])
    m2.metric("Benchmarks Found", metrics["matched_rows"])
    m3.metric("Stable Baseline Rows", f"{metrics['stable_rows']} / {metrics['total_clean']}")
    match_rate = round((metrics["matched_rows"] / metrics["test_rows"]) * 100, 1) if metrics["test_rows"] > 0 else 0
    m4.metric("Match Success Rate", f"{match_rate}%")

    st.download_button(
        label="⬇️ Download Output Excel Report (.xlsx — Sheet 1: lbt_input | Sheet 2: Results | Sheet 3: LinePlot)",
        data=st.session_state["excel_bytes"],
        file_name="Cryo_2_LBT_Result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    tab_table, tab_plot, tab_cfg = st.tabs([
        "📋 Optimal Benchmark Table (Results)",
        "📈 Match Tag Iterations Plot (LinePlot)",
        "⚙️ Applied Config Snapshot (lbt_input)",
    ])

    with tab_table:
        st.dataframe(final_df, use_container_width=True, height=450)

    with tab_plot:
        if st.session_state["fig"] is not None:
            st.pyplot(st.session_state["fig"])
        else:
            st.info("No iteration columns found to generate the line plot.")

    with tab_cfg:
        st.dataframe(st.session_state["used_L_df"], use_container_width=True)
