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
        raise ValueError("No valid 'Iteration No' values found in lbt_input configuration.")
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
    if "performance_tag" not in L.columns or "direction" not in L.columns:
        st.warning("⚠️ 'performance_tag' or 'direction' column missing in config - skipping optimal filtering.")
        return df

    perf_config = L[["performance_tag", "direction"]].dropna().drop_duplicates()
    if perf_config.empty:
        st.warning("⚠️ No performance tag found in config - skipping filtering.")
        return df

    perf_tag = str(perf_config.iloc[0]["performance_tag"]).strip()
    direction = str(perf_config.iloc[0]["direction"]).strip().lower()
    perf_col = f"{perf_tag}_benchmark"
    if perf_col not in df.columns:
        st.warning(f"⚠️ Performance column '{perf_col}' not found - skipping filtering.")
        return df

    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")
    matched_df = df.dropna(subset=[perf_col])
    unmatched_df = df[df[perf_col].isna()]

    if direction == "minimize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmin()]
    elif direction == "maximize":
        optimal_df = matched_df.loc[matched_df.groupby("Timestamp_Actual")[perf_col].idxmax()]
    else:
        raise ValueError(f"Unknown direction '{direction}' in performance config. Must be 'minimize' or 'maximize'.")

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

    # Write Results + Used LBT Config to in-memory Excel file
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
        used_L_df.to_excel(writer, index=False, sheet_name="lbt_input_used")

    # Embed image into Excel if plot was created
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
st.markdown("Upload your process data workbook, tweak tolerances or KPI parameters live in the table below, and run instant benchmarking.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Global Settings")

    st.subheader("1. Stability Filter")
    feed_tag = st.text_input("Feed Column Tag", value="feed_MMSCFD")
    col1, col2 = st.columns(2)
    with col1:
        feed_min = st.number_input("Min Feed", value=180.0, step=5.0)
    with col2:
        feed_max = st.number_input("Max Feed", value=250.0, step=5.0)

    st.subheader("2. Matching Settings")
    max_benchmarks = st.number_input("Max Benchmarks / Match", min_value=1, max_value=100, value=10, step=1)

    with st.expander("Sheet Names (Advanced)"):
        sheet_t = st.text_input("Test Data Sheet", value="test_data")
        sheet_c = st.text_input("Clean Data Sheet", value="clean_data")
        sheet_l = st.text_input("LBT Config Sheet", value="lbt_input")

# --- Main File Uploader ---
uploaded_file = st.file_uploader("📥 Upload Input Excel File (.xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Track file identity so we only reset the editor when a NEW file is uploaded
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("current_file_id") != file_id:
        try:
            T_raw, C_raw, L_raw = load_sheets(uploaded_file, sheet_t, sheet_c, sheet_l)
            st.session_state["current_file_id"] = file_id
            st.session_state["T_raw"] = T_raw
            st.session_state["C_raw"] = C_raw
            st.session_state["L_original"] = L_raw.copy()
            st.session_state["L_working"] = L_raw.copy()
            # Clear previous results when a brand-new file is loaded
            st.session_state.pop("final_df", None)
        except Exception as e:
            st.error(f"❌ Could not read sheets from Excel file: {e}")

if "L_working" in st.session_state:
    st.divider()
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.subheader("🛠️ Live LBT Configuration Editor (`lbt_input`)")
        st.caption("Edit tolerances, add/delete iteration rows (`+` at bottom), or change the performance tag & direction below.")
    with header_col2:
        if st.button("↺ Reset to Original Excel", use_container_width=True):
            st.session_state["L_working"] = st.session_state["L_original"].copy()
            st.rerun()

    # Quick Performance Tag & Direction override bar
    L_current = st.session_state["L_working"].copy()
    available_tags = [c for c in st.session_state["C_raw"].columns if c != "Timestamp"]

    if "performance_tag" in L_current.columns and "direction" in L_current.columns:
        valid_perf = L_current[["performance_tag", "direction"]].dropna()
        default_perf = str(valid_perf.iloc[0]["performance_tag"]).strip() if not valid_perf.empty else (available_tags[0] if available_tags else "")
        default_dir = str(valid_perf.iloc[0]["direction"]).strip().lower() if not valid_perf.empty else "minimize"

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if default_perf in available_tags:
                selected_perf = st.selectbox("🎯 Target Performance Tag (KPI)", options=available_tags, index=available_tags.index(default_perf))
            else:
                selected_perf = st.text_input("🎯 Target Performance Tag (KPI)", value=default_perf)
        with p_col2:
            dir_options = ["minimize", "maximize"]
            dir_idx = dir_options.index(default_dir) if default_dir in dir_options else 0
            selected_dir = st.selectbox("🧭 Optimization Direction", options=dir_options, index=dir_idx)

        # Sync quick-selector changes back into the working dataframe
        L_current["performance_tag"] = np.nan
        L_current["direction"] = np.nan
        if len(L_current) > 0:
            L_current.loc[L_current.index[0], "performance_tag"] = selected_perf
            L_current.loc[L_current.index[0], "direction"] = selected_dir

    # Interactive Editable Table
    edited_L = st.data_editor(
        L_current,
        num_rows="dynamic",
        use_container_width=True,
        key="lbt_editor",
    )

    # Run Button
    if st.button("▶ Run Benchmarking Analysis", type="primary", use_container_width=True):
        with st.spinner("Running analysis with your live configuration..."):
            try:
                T = st.session_state["T_raw"].copy()
                C = st.session_state["C_raw"].copy()
                L_active = edited_L.copy()

                # Ensure performance_tag & direction persist even if user added rows
                if "performance_tag" in L_active.columns and "direction" in L_active.columns and len(L_active) > 0:
                    L_active["performance_tag"] = np.nan
                    L_active["direction"] = np.nan
                    L_active.loc[L_active.index[0], "performance_tag"] = selected_perf
                    L_active.loc[L_active.index[0], "direction"] = selected_dir

                # Build decimal dictionary from edited table
                valid_dec = L_active[["Match_tags", "Match_Tag_Decimal"]].dropna()
                decimal_dict = valid_dec.set_index("Match_tags")["Match_Tag_Decimal"].to_dict()

                total_clean_rows = len(C)

                # 1. Stability Filter
                C = add_feed_stability(C, feed_col=feed_tag.strip(), min_val=feed_min, max_val=feed_max)
                if "Feed_Stable" in C.columns:
                    C = C[C["Feed_Stable"] == "Stable"].copy()
                stable_clean_rows = len(C)

                # 2. Match & Optimize
                results = match_records(T, C, L_active, int(max_benchmarks), decimal_dict)
                final_df = process_results(results, T, C, L_active)

                # 3. Generate Plot & Output Excel in Memory
                excel_bytes, fig = generate_plot_and_excel_bytes(final_df, L_active)

                st.session_state["final_df"] = final_df
                st.session_state["excel_bytes"] = excel_bytes
                st.session_state["fig"] = fig
                st.session_state["metrics"] = {
                    "test_rows": len(T),
                    "stable_rows": stable_clean_rows,
                    "total_clean": total_clean_rows,
                    "matched_rows": int((final_df["iteration_matched"] != "BENCHMARK NOT FOUND").sum())
                    if "iteration_matched" in final_df.columns else 0,
                }
                st.success("✅ Benchmarking analysis completed with your live settings!")

            except Exception as e:
                st.error(f"❌ Error during execution: {e}")

# --- Display Results if Available ---
if "final_df" in st.session_state:
    st.divider()
    metrics = st.session_state["metrics"]
    final_df = st.session_state["final_df"]

    # KPI Summary Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Records Evaluated", metrics["test_rows"])
    m2.metric("Benchmarks Found", metrics["matched_rows"])
    m3.metric("Stable Baseline Rows", f"{metrics['stable_rows']} / {metrics['total_clean']}")
    match_rate = round((metrics["matched_rows"] / metrics["test_rows"]) * 100, 1) if metrics["test_rows"] > 0 else 0
    m4.metric("Match Success Rate", f"{match_rate}%")

    # Download Button
    st.download_button(
        label="⬇️ Download Output Excel Report (.xlsx with Results, Config & LinePlot)",
        data=st.session_state["excel_bytes"],
        file_name="Cryo_2_LBT_Result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    # Tabs for Data Table and Plot
    tab_table, tab_plot = st.tabs(["📊 Optimal Benchmark Results", "📈 Match Tag Iterations Plot"])

    with tab_table:
        st.dataframe(final_df, use_container_width=True, height=450)

    with tab_plot:
        if st.session_state["fig"] is not None:
            st.pyplot(st.session_state["fig"])
        else:
            st.info("No iteration columns found to generate the line plot.")
