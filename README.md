# ⚡ Look-Back Benchmarking (LBT) Studio

A web-based process engineering tool for automated historical look-back benchmarking, feed stability filtering, and KPI optimization.

## 📋 Required Input Excel Format (`.xlsx`)
Your uploaded Excel workbook must contain three sheets (customizable in the app's sidebar):
1. **`test_data`**: Current operating measurements with a `Timestamp` column and process tags.
2. **`clean_data`**: Historical baseline records containing `Timestamp`, process tags, and your feed rate column (e.g., `feed_MMSCFD`).
3. **`lbt_input`**: Configuration table with columns:
   - `Match_tags`
   - `Match_Tag_Decimal`
   - `Iteration No`
   - `Tolerance_minimum`
   - `Tolerance_maximum`
   - `performance_tag`
   - `direction` (`minimize` or `maximize`)

## 🚀 How to Use
1. Open the live web app link.
2. Set your **Feed Column Tag** and **Min/Max Stable Feed** limits in the left sidebar.
3. Upload your `.xlsx` file and click **Run Benchmarking Analysis**.
4. Review the matched results and iteration plot, then download the generated `.xlsx` report.
