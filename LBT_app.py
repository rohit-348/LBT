import io
import os
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

def load_data(uploaded_file, sheet_t, sheet_c, sheet_l):
    T = pd.read_excel(uploaded_file, sheet_name=sheet_t)
    C = pd.read_excel(uploaded_file, sheet_name=sheet_c)
    L = pd.read_excel(uploaded_file, sheet_name=sheet_l)

    T["Timestamp"] = pd.to_datetime(T["Timestamp"], errors="coerce")
    C["Timestamp"] = pd.to_datetime(C["Timestamp"], errors="coerce")

    decimal_dict = L.set_index("Match_tags")["Match_Tag_Decimal"].dropna().to_dict()
    return T, C, L, decimal_dict


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
                if -min_tol <= delta <= max_tol:
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
    max_iterations = int(pd.to_numeric(L["Iteration No"], errors="coerce").dropna().max())

    for _, row_t in T.iterrows():
        match_found = False
        for iteration in range(1, max_iterations + 1):
            current_tol = L[L["Iteration No"] == iteration]
            filtered_C = C.copy()
            for _, tol_row in current_tol.iterrows():
                col = tol_row["Match_tags"]
                min_tol = tol_row["Tolerance_minimum"]
                max_tol = tol_row["Tolerance_maximum"]
                if col not in T.columns or col not in C.columns or col == "Timestamp":
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
        st.warning("⚠️ No performance tag found in config - skipping filtering.")
        return df

    perf_tag = perf_config.iloc[0]["performance_tag"]
    direction = perf_config.iloc[0]["direction"].strip().lower()
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


def generate_plot_and_excel_bytes(df, plot_sheet_name="LinePlot"):
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

    # Write DataFrame to in-memory Excel file
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

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
st.markdown("Upload your process data workbook to filter stable operating periods, match historical tolerances, and export optimal benchmarks.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")

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
    if st.button("▶ Run Benchmarking Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing data and evaluating tolerance iterations..."):
            try:
                # 1. Load Data
                T, C, L, decimal_dict = load_data(uploaded_file, sheet_t, sheet_c, sheet_l)
                total_clean_rows = len(C)

                # 2. Stability Filter
                C = add_feed_stability(C, feed_col=feed_tag.strip(), min_val=feed_min, max_val=feed_max)
                if "Feed_Stable" in C.columns:
                    C = C[C["Feed_Stable"] == "Stable"].copy()
                stable_clean_rows = len(C)

                # 3. Match & Optimize
                results = match_records(T, C, L, int(max_benchmarks), decimal_dict)
                final_df = process_results(results, T, C, L)

                # 4. Generate Plot & Output Excel in Memory
                excel_bytes, fig = generate_plot_and_excel_bytes(final_df)

                # Save outputs in session_state so they persist across interactions
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
                st.success("✅ Benchmarking analysis completed successfully!")

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
        label="⬇️ Download Output Excel Report (.xlsx with LinePlot)",
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