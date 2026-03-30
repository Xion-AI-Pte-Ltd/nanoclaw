#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell


OPERATING_TOKENS = [
    "revenue",
    "sales",
    "income",
    "expense",
    "cost of sales",
    "accounts receivable",
    "accounts payable",
    "inventory",
    "tax",
    "payroll",
    "salary",
    "wages",
]
INVESTING_TOKENS = [
    "fixed asset",
    "property",
    "plant",
    "equipment",
    "intangible",
    "vehicle",
    "renovation",
    "acquisition",
    "disposal",
    "capital expenditure",
    "capex",
    "investment",
]
FINANCING_TOKENS = [
    "loan",
    "borrow",
    "equity",
    "share",
    "capital",
    "dividend",
    "repayment",
    "financing",
]
BANK_TOKENS = ["bank", "cash", "checking", "current account", "petty cash"]


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def safe_float(value: Any) -> float:
    text = normalize_text(value)
    if not text:
        return 0.0
    text = text.replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    if text and text[0] in {"$", "€", "£"}:
        text = text[1:]
    try:
        return float(text)
    except Exception:
        return 0.0


def truthy_cash_entry(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value)
    return text in {"1", "true", "yes", "y", "t", "cash", "cash entry"}


def parse_date(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    patterns = (
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%a, %d %b %Y %H:%M:%S %Z",
    )
    for pattern in patterns:
        try:
            return datetime.strptime(text, pattern)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def parse_month(value: Any) -> str:
    parsed = parse_date(value)
    return parsed.strftime("%Y-%m") if parsed else "Unknown"


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def classification_hints(value: Any) -> list[str]:
    hints: list[str] = []
    if isinstance(value, str):
        hints.append(value)
    elif isinstance(value, dict):
        for key in (
            "cash_flow_activity",
            "CashFlowActivity",
            "activity",
            "Activity",
            "statement_type",
            "StatementType",
            "category",
            "Category",
            "account_type",
            "AccountType",
            "type",
            "Type",
            "group",
            "Group",
        ):
            item = value.get(key)
            if item not in (None, ""):
                hints.append(str(item))
    return hints


def infer_activity(*, account_name: Any, account_type: Any, description: Any, coa_value: Any) -> str:
    hints = [
        normalize_text(account_name),
        normalize_text(account_type),
        normalize_text(description),
        *[normalize_text(h) for h in classification_hints(coa_value)],
    ]
    combined = " ".join(h for h in hints if h).lower()
    if any(token in combined for token in BANK_TOKENS):
        return "Operating"
    if any(token in combined for token in FINANCING_TOKENS):
        return "Financing"
    if any(token in combined for token in INVESTING_TOKENS):
        return "Investing"
    if any(token in combined for token in OPERATING_TOKENS):
        return "Operating"
    return "Operating"


def build_coa_lookup(raw: Any) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(key, str) and key.strip():
                lookup[key.strip().lower()] = value
        return lookup
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            for key in ("Account", "account", "AccountName", "account_name", "name", "Name"):
                account_name = item.get(key)
                if isinstance(account_name, str) and account_name.strip():
                    lookup[account_name.strip().lower()] = item
                    break
    return lookup


def maybe_find_coa_value(lookup: dict[str, Any], account_name: Any) -> Any:
    key = normalize_text(account_name)
    if not key:
        return None
    if key in lookup:
        return lookup[key]
    for candidate, value in lookup.items():
        if candidate == key or candidate in key or key in candidate:
            return value
    return None


def coa_value_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("cash_flow_activity", "CashFlowActivity", "activity", "Activity", "category", "Category", "AccountType", "account_type", "type", "Type"):
            item = value.get(key)
            if item not in (None, ""):
                return str(item).strip()
    return ""


def round_currency(value: float) -> float:
    return round(float(value or 0.0), 2)


def normalize_zero(value: float) -> float:
    return 0.0 if abs(float(value or 0.0)) < 0.005 else round_currency(value)


def month_sort_key(month: str) -> tuple[int, int]:
    try:
        year, month_num = month.split("-", 1)
        return int(year), int(month_num)
    except Exception:
        return (9999, 99)


def month_endpoints(month: str) -> datetime | None:
    try:
        return datetime.strptime(f"{month}-01", "%Y-%m-%d")
    except Exception:
        return None


def iter_month_series(start_month: str, end_month: str) -> list[str]:
    start_dt = month_endpoints(start_month)
    end_dt = month_endpoints(end_month)
    if not start_dt or not end_dt:
        return sorted({start_month, end_month}, key=month_sort_key)
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    series: list[str] = []
    current = start_dt
    while current <= end_dt:
        series.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)
    return series


def write_headers(ws: xlsxwriter.workbook.Worksheet, row: int, headers: list[str], header_fmt: Any) -> None:
    for col, header in enumerate(headers):
        ws.write(row, col, header, header_fmt)


def cell_ref(row: int, col: int) -> str:
    return xl_rowcol_to_cell(row, col, row_abs=True, col_abs=True)


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: cash_flow_variance_workbook.py <bundle_path> <output_path>")

    bundle_path = Path(sys.argv[1]).expanduser().resolve()
    output_path = Path(sys.argv[2]).expanduser().resolve()
    if not bundle_path.exists():
        raise SystemExit(f"Bundle file not found: {bundle_path}")

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    datasets = bundle.get("datasets_by_key", {}) if isinstance(bundle, dict) else {}
    gl_block = datasets.get("general_ledger", {}) if isinstance(datasets, dict) else {}
    gl_result = gl_block.get("response", {}).get("result", {}) if isinstance(gl_block, dict) else {}
    gl_rows = as_dict_list(gl_result.get("gl")) if isinstance(gl_result, dict) else []
    coa_block = datasets.get("chart_of_accounts", {}) if isinstance(datasets, dict) else {}
    coa_result = coa_block.get("response", {}).get("result", {}) if isinstance(coa_block, dict) else {}
    trial_balance = datasets.get("trial_balance", {}) if isinstance(datasets, dict) else {}
    report_request = bundle.get("report_request", {}) if isinstance(bundle, dict) else {}
    if not isinstance(report_request, dict):
        report_request = {}

    coa_lookup = build_coa_lookup(coa_result)
    tx_groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in gl_rows:
        if not truthy_cash_entry(row.get("Cash Entry")):
            continue
        match_id = str(row.get("MatchID") or row.get("match_id") or row.get("EntryID") or "").strip()
        if not match_id:
            match_id = f"row-{len(tx_groups) + 1}"
        tx_groups.setdefault(match_id, []).append(row)

    processed_rows: list[dict[str, Any]] = []
    monthly: OrderedDict[str, dict[str, float]] = OrderedDict()
    activity_totals = {"Operating": 0.0, "Investing": 0.0, "Financing": 0.0}
    account_summaries: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for match_id, group_rows in tx_groups.items():
        if not group_rows:
            continue
        cash_rows = [
            row
            for row in group_rows
            if any(token in normalize_text(row.get("Account")) for token in BANK_TOKENS)
            or "asset" in normalize_text(row.get("AccountType"))
            or "cash" in normalize_text(row.get("Account"))
        ]
        cash_row = cash_rows[0] if cash_rows else max(
            group_rows,
            key=lambda row: abs(safe_float(row.get("Amount"))) or abs(safe_float(row.get("Credit"))) + abs(safe_float(row.get("Debit"))),
        )
        counterpart_rows = [row for row in group_rows if row is not cash_row]
        reference_row = counterpart_rows[0] if counterpart_rows else cash_row

        month = parse_month(cash_row.get("Date") or reference_row.get("Date"))
        debit = safe_float(cash_row.get("Debit"))
        credit = safe_float(cash_row.get("Credit"))
        amount = safe_float(cash_row.get("Amount"))
        if abs(amount) < 0.0001:
            amount = safe_float(cash_row.get("Credit")) - safe_float(cash_row.get("Debit"))
        net = round_currency(amount)
        cash_account = str(cash_row.get("Account") or cash_row.get("Name") or "").strip()
        counterparty_account = str(reference_row.get("Account") or reference_row.get("Name") or "").strip()
        account_type = str(reference_row.get("AccountType") or cash_row.get("AccountType") or "").strip()
        description = str(cash_row.get("Description") or reference_row.get("Description") or "").strip()
        coa_value = maybe_find_coa_value(coa_lookup, counterparty_account or cash_account)
        activity = infer_activity(
            account_name=counterparty_account or cash_account,
            account_type=account_type,
            description=description,
            coa_value=coa_value,
        )

        processed_rows.append(
            {
                "MatchID": match_id,
                "Date": str(cash_row.get("Date") or reference_row.get("Date") or "").strip(),
                "Month": month,
                "CashAccount": cash_account,
                "CounterpartAccount": counterparty_account,
                "AccountType": account_type,
                "Description": description,
                "Debit": round_currency(debit),
                "Credit": round_currency(credit),
                "CashMovement": net,
                "Activity": activity,
                "CashEntry": True,
            }
        )

        monthly.setdefault(
            month,
            {
                "Operating": 0.0,
                "Investing": 0.0,
                "Financing": 0.0,
                "Net Cash Flow": 0.0,
                "Cash Rows": 0.0,
            },
        )
        monthly[month][activity] += net
        monthly[month]["Net Cash Flow"] += net
        monthly[month]["Cash Rows"] += 1
        activity_totals[activity] += net

        summary = account_summaries.setdefault(
            counterparty_account or cash_account,
            {
                "Account": counterparty_account or cash_account,
                "AccountType": account_type,
                "Classification": coa_value_text(coa_value) or account_type or "Unknown",
                "Activity": activity,
                "NetCashFlow": 0.0,
                "CashRows": 0,
            },
        )
        summary["NetCashFlow"] += net
        summary["CashRows"] += 1

    months = sorted(monthly.keys(), key=month_sort_key)
    time_window = report_request.get("time_window") if isinstance(report_request.get("time_window"), dict) else {}
    requested_end_month = ""
    if isinstance(time_window, dict):
        requested_end = str(time_window.get("end_date") or time_window.get("end") or "").strip()
        requested_end_month = parse_month(requested_end) if requested_end else ""
    if months:
        end_month = requested_end_month or months[-1]
        months = iter_month_series(months[0], end_month)
        for month in months:
            monthly.setdefault(
                month,
                {
                    "Operating": 0.0,
                    "Investing": 0.0,
                    "Financing": 0.0,
                    "Net Cash Flow": 0.0,
                    "Cash Rows": 0.0,
                },
            )
    variance_rows: list[dict[str, Any]] = []
    for index, month in enumerate(months):
        actual = round_currency(monthly[month]["Net Cash Flow"])
        history_months = months[max(0, index - 3):index]
        history_values = [monthly[m]["Net Cash Flow"] for m in history_months]
        baseline = round_currency(sum(history_values) / len(history_values)) if history_values else None
        variance = round_currency(actual - baseline) if baseline is not None else None
        variance_pct = variance / baseline if baseline not in (None, 0) and variance is not None else None
        threshold = max(abs(baseline or 0.0) * 0.2, 2000.0) if baseline is not None else 2000.0
        outlier = bool(variance is not None and abs(variance) > threshold)
        variance_rows.append(
            {
                "Month": month,
                "Actual": actual,
                "3M Rolling Avg": baseline,
                "Variance": variance,
                "Variance %": variance_pct,
                "Outlier": "Yes" if outlier else "No",
                "Comment": "Historical average baseline" if baseline is not None else "Insufficient history",
            }
        )

    total_cash_flow = normalize_zero(sum(item["Net Cash Flow"] for item in monthly.values()))
    average_monthly_cash_flow = round_currency(total_cash_flow / len(months)) if months else 0.0
    max_month = max(months, key=lambda month: monthly[month]["Net Cash Flow"]) if months else ""
    min_month = min(months, key=lambda month: monthly[month]["Net Cash Flow"]) if months else ""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(output_path), {"nan_inf_to_errors": True})

    title_fmt = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#16324f"})
    subtitle_fmt = workbook.add_format({"italic": True, "font_color": "#475569"})
    section_fmt = workbook.add_format({"bold": True, "bg_color": "#dbeafe", "font_color": "#0f172a"})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#1d4ed8", "font_color": "#ffffff", "border": 1})
    text_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    money_fmt = workbook.add_format({"num_format": '#,##0.00;[Red]-#,##0.00'})
    pct_fmt = workbook.add_format({"num_format": '0.0%;[Red]-0.0%'})
    integer_fmt = workbook.add_format({"num_format": "0"})
    note_fmt = workbook.add_format({"text_wrap": True, "font_color": "#475569"})
    flag_yes_fmt = workbook.add_format({"bg_color": "#fee2e2", "font_color": "#991b1b"})
    flag_no_fmt = workbook.add_format({"bg_color": "#dcfce7", "font_color": "#166534"})

    # Executive Summary
    ws = workbook.add_worksheet("Executive Summary")
    ws.set_column("A:A", 26)
    ws.set_column("B:F", 18)
    ws.merge_range("A1:F1", "Cash Flow Variance Analysis", title_fmt)
    ws.write("A2", "Derived from the staged general ledger bundle; cash-flow transactions are collapsed by MatchID and classified using the chart of accounts when available.", subtitle_fmt)
    ws.write("A3", f"Source bundle: {bundle_path.name}", subtitle_fmt)
    time_window = report_request.get("time_window") if isinstance(report_request.get("time_window"), dict) else {}
    if isinstance(time_window, dict):
        start = str(time_window.get("start_date") or time_window.get("start") or "").strip()
        end = str(time_window.get("end_date") or time_window.get("end") or "").strip()
    else:
        start = ""
        end = ""
    ws.write("A4", f"Period: {start or 'n/a'} to {end or 'n/a'}", subtitle_fmt)

    data_header_row = 3
    data_first_row = data_header_row + 1
    data_first_excel_row = data_first_row + 1
    data_last_excel_row = data_first_excel_row + len(processed_rows) - 1

    metrics = [
        ("Cash transactions processed", len(processed_rows)),
        ("Months analysed", len(months)),
        ("Total net cash flow", total_cash_flow),
        ("Average monthly net cash flow", average_monthly_cash_flow),
        ("Operating total", round_currency(activity_totals["Operating"])),
        ("Investing total", round_currency(activity_totals["Investing"])),
        ("Financing total", round_currency(activity_totals["Financing"])),
    ]
    monthly_start_row = 8 + len(metrics) + 2
    ws.write("A6", "Key Metrics", section_fmt)
    write_headers(ws, 7, ["Metric", "Value"], header_fmt)
    for idx, (label, value) in enumerate(metrics, start=8):
        ws.write(idx, 0, label)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "rows" in label.lower() or "months" in label.lower():
                ws.write_number(idx, 1, float(value), integer_fmt)
            else:
                ws.write_number(idx, 1, float(value), money_fmt)
        else:
            ws.write(idx, 1, value)

    metric_value_row = 8
    if processed_rows:
        ws.write_formula(
            metric_value_row,
            1,
            f"=COUNTA('Data Source'!$A${data_first_excel_row}:$A${data_last_excel_row})",
            integer_fmt,
            len(processed_rows),
        )
    else:
        ws.write_number(metric_value_row, 1, 0, integer_fmt)
    if months:
        ws.write_formula(
            metric_value_row + 1,
            1,
            f"=COUNTA(A{monthly_start_row + 2}:A{monthly_start_row + 1 + len(months)})",
            integer_fmt,
            len(months),
        )
    else:
        ws.write_number(metric_value_row + 1, 1, 0, integer_fmt)
    ws.write(monthly_start_row - 1, 0, "Monthly Cash Flow Trend", section_fmt)
    monthly_headers = ["Month", "Operating", "Investing", "Financing", "Net Cash Flow", "3M Rolling Avg"]
    write_headers(ws, monthly_start_row, monthly_headers, header_fmt)
    for offset, month in enumerate(months, start=monthly_start_row + 1):
        row = monthly[month]
        summary_row_index = offset
        month_cell = cell_ref(summary_row_index, 0)
        operating_formula = (
            f'=SUMIFS(\'Data Source\'!$J${data_first_excel_row}:$J${data_last_excel_row},'
            f'\'Data Source\'!$C${data_first_excel_row}:$C${data_last_excel_row},{month_cell},'
            f'\'Data Source\'!$K${data_first_excel_row}:$K${data_last_excel_row},"Operating")'
        )
        investing_formula = (
            f'=SUMIFS(\'Data Source\'!$J${data_first_excel_row}:$J${data_last_excel_row},'
            f'\'Data Source\'!$C${data_first_excel_row}:$C${data_last_excel_row},{month_cell},'
            f'\'Data Source\'!$K${data_first_excel_row}:$K${data_last_excel_row},"Investing")'
        )
        financing_formula = (
            f'=SUMIFS(\'Data Source\'!$J${data_first_excel_row}:$J${data_last_excel_row},'
            f'\'Data Source\'!$C${data_first_excel_row}:$C${data_last_excel_row},{month_cell},'
            f'\'Data Source\'!$K${data_first_excel_row}:$K${data_last_excel_row},"Financing")'
        )
        net_formula = (
            f'=SUMIFS(\'Data Source\'!$J${data_first_excel_row}:$J${data_last_excel_row},'
            f'\'Data Source\'!$C${data_first_excel_row}:$C${data_last_excel_row},{month_cell})'
        )
        previous_cells = [
            cell_ref(summary_row_index - delta, 4)
            for delta in range(1, min(3, summary_row_index - (monthly_start_row + 1)) + 1)
        ]
        if previous_cells:
            rolling_formula = "=" + (previous_cells[0] if len(previous_cells) == 1 else f"AVERAGE({','.join(previous_cells)})")
            history_values = [monthly[m]["Net Cash Flow"] for m in months[max(0, months.index(month) - 3):months.index(month)]]
            baseline = round_currency(sum(history_values) / len(history_values)) if history_values else None
        else:
            rolling_formula = ""
            baseline = None
        ws.write(offset, 0, month)
        ws.write_formula(offset, 1, operating_formula, money_fmt, round_currency(row["Operating"]))
        ws.write_formula(offset, 2, investing_formula, money_fmt, round_currency(row["Investing"]))
        ws.write_formula(offset, 3, financing_formula, money_fmt, round_currency(row["Financing"]))
        ws.write_formula(offset, 4, net_formula, money_fmt, round_currency(row["Net Cash Flow"]))
        if baseline is not None:
            ws.write_formula(offset, 5, rolling_formula, money_fmt, baseline)
        else:
            ws.write_blank(offset, 5, None)

    if months:
        chart = workbook.add_chart({"type": "line"})
        chart.add_series(
            {
                "name": "Net Cash Flow",
                "categories": ["Executive Summary", monthly_start_row + 1, 0, monthly_start_row + len(months), 0],
                "values": ["Executive Summary", monthly_start_row + 1, 4, monthly_start_row + len(months), 4],
                "line": {"color": "#2563eb", "width": 2.25},
            }
        )
        chart.add_series(
            {
                "name": "3M Rolling Avg",
                "categories": ["Executive Summary", monthly_start_row + 1, 0, monthly_start_row + len(months), 0],
                "values": ["Executive Summary", monthly_start_row + 1, 5, monthly_start_row + len(months), 5],
                "line": {"color": "#7c3aed", "dash_type": "dash", "width": 2},
            }
        )
        chart.set_title({"name": "Monthly Net Cash Flow vs Rolling Average"})
        chart.set_legend({"position": "bottom"})
        chart.set_size({"width": 760, "height": 320})
        ws.insert_chart(monthly_start_row + len(months) + 2, 0, chart)

    # Variance Analysis
    variance_ws = workbook.add_worksheet("Variance Analysis")
    variance_ws.set_column("A:A", 14)
    variance_ws.set_column("B:E", 18)
    variance_ws.set_column("F:G", 20)
    variance_ws.merge_range("A1:G1", "Variance Analysis", title_fmt)
    variance_ws.write("A2", "One row per month using a trailing 3-month rolling average baseline.", subtitle_fmt)
    variance_headers = ["Month", "Actual", "3M Rolling Avg", "Variance", "Variance %", "Outlier", "Comment"]
    write_headers(variance_ws, 3, variance_headers, header_fmt)
    for idx, row in enumerate(variance_rows, start=4):
        summary_row_idx = monthly_start_row + 1 + (idx - 4)
        actual_cell = cell_ref(summary_row_idx, 4)
        rolling_cell = cell_ref(summary_row_idx, 5)
        variance_formula = f"={cell_ref(idx, 1)}-{cell_ref(idx, 2)}"
        variance_pct_formula = f'=IF({cell_ref(idx, 2)}=0,"",{cell_ref(idx, 3)}/{cell_ref(idx, 2)})'
        outlier_formula = f'=IF(ABS({cell_ref(idx, 3)})>MAX(ABS({cell_ref(idx, 2)})*0.2,2000),"Yes","No")'
        comment_formula = f'=IF({cell_ref(idx, 2)}="","Insufficient history","Historical average baseline")'
        variance_ws.write(idx, 0, row["Month"])
        variance_ws.write_formula(idx, 1, f"='Executive Summary'!{actual_cell}", money_fmt, float(row["Actual"]))
        if row["3M Rolling Avg"] is not None:
            variance_ws.write_formula(idx, 2, f"='Executive Summary'!{rolling_cell}", money_fmt, float(row["3M Rolling Avg"]))
        else:
            variance_ws.write_blank(idx, 2, None)
        if row["Variance"] is not None:
            variance_ws.write_formula(idx, 3, variance_formula, money_fmt, float(row["Variance"]))
        else:
            variance_ws.write_blank(idx, 3, None)
        if row["Variance %"] is not None:
            variance_ws.write_formula(idx, 4, variance_pct_formula, pct_fmt, float(row["Variance %"]))
        else:
            variance_ws.write_blank(idx, 4, None)
        outlier = row["Outlier"]
        variance_ws.write_formula(idx, 5, outlier_formula, flag_yes_fmt if outlier == "Yes" else flag_no_fmt, outlier)
        variance_ws.write_formula(idx, 6, comment_formula, note_fmt, row["Comment"])
    if variance_rows:
        variance_chart = workbook.add_chart({"type": "line"})
        variance_chart.add_series(
            {
                "name": "Actual",
                "categories": ["Variance Analysis", 4, 0, 3 + len(variance_rows), 0],
                "values": ["Variance Analysis", 4, 1, 3 + len(variance_rows), 1],
                "line": {"color": "#0f766e", "width": 2.25},
            }
        )
        variance_chart.add_series(
            {
                "name": "3M Rolling Avg",
                "categories": ["Variance Analysis", 4, 0, 3 + len(variance_rows), 0],
                "values": ["Variance Analysis", 4, 2, 3 + len(variance_rows), 2],
                "line": {"color": "#b45309", "dash_type": "dash", "width": 2},
            }
        )
        variance_chart.set_title({"name": "Actual vs Rolling Average"})
        variance_chart.set_legend({"position": "bottom"})
        variance_chart.set_size({"width": 760, "height": 320})
        variance_ws.insert_chart(4 + len(variance_rows) + 2, 0, variance_chart)

    # Data Source
    data_ws = workbook.add_worksheet("Data Source")
    data_ws.set_column("A:A", 28)
    data_ws.set_column("B:B", 12)
    data_ws.set_column("C:C", 24)
    data_ws.set_column("D:D", 22)
    data_ws.set_column("E:E", 40)
    data_ws.set_column("F:H", 16)
    data_ws.set_column("I:I", 14)
    data_ws.set_column("J:J", 12)
    data_ws.merge_range("A1:J1", "Processed Cash-Flow Rows", title_fmt)
    data_ws.write("A2", "Source: general_ledger rows with a truthy Cash Entry flag.", subtitle_fmt)
    processed_headers = [
        "MatchID",
        "Date",
        "Month",
        "CashAccount",
        "CounterpartAccount",
        "AccountType",
        "Description",
        "Debit",
        "Credit",
        "CashMovement",
        "Activity",
        "CashEntry",
    ]
    start_row = 3
    write_headers(data_ws, start_row, processed_headers, header_fmt)
    for idx, row in enumerate(processed_rows, start=start_row + 1):
        data_ws.write(idx, 0, row["MatchID"])
        data_ws.write(idx, 1, row["Date"])
        data_ws.write(idx, 2, row["Month"])
        data_ws.write(idx, 3, row["CashAccount"])
        data_ws.write(idx, 4, row["CounterpartAccount"])
        data_ws.write(idx, 5, row["AccountType"])
        data_ws.write(idx, 6, row["Description"], text_fmt)
        data_ws.write_number(idx, 7, float(row["Debit"]), money_fmt)
        data_ws.write_number(idx, 8, float(row["Credit"]), money_fmt)
        data_ws.write_number(idx, 9, float(row["CashMovement"]), money_fmt)
        data_ws.write(idx, 10, row["Activity"])
        data_ws.write_boolean(idx, 11, bool(row["CashEntry"]))

    coa_row = start_row + max(1, len(processed_rows)) + 3
    data_ws.write(coa_row, 0, "Chart of Accounts Mapping", section_fmt)
    coa_headers = ["Account", "Classification", "Inferred Activity", "Source"]
    write_headers(data_ws, coa_row + 1, coa_headers, header_fmt)
    coa_entries_written = 0
    for idx, (account, value) in enumerate(sorted(coa_lookup.items()), start=coa_row + 2):
        classification = coa_value_text(value) or "Unknown"
        activity = infer_activity(
            account_name=account,
            account_type=classification,
            description="",
            coa_value=value,
        )
        data_ws.write(idx, 0, account)
        data_ws.write(idx, 1, classification)
        data_ws.write(idx, 2, activity)
        data_ws.write(idx, 3, "chart_of_accounts")
        coa_entries_written += 1

    # Assumptions & Logic
    logic_ws = workbook.add_worksheet("Assumptions & Logic")
    logic_ws.set_column("A:A", 110)
    logic_ws.write("A1", "Assumptions & Logic", title_fmt)
    assumptions = [
        "Cash-flow events are derived by grouping general_ledger entries on MatchID and selecting the cash-account side of each pair.",
        "Monthly buckets use the parsed cash-row Date and are normalized to YYYY-MM.",
        "Cash movement uses the cash-row Amount when available, falling back to Credit - Debit only when Amount is unavailable.",
        "Key report sheets are formula-driven so totals and variances can be traced directly from the Data Source sheet.",
        "Classification uses chart_of_accounts on the counterpart account when available; otherwise activity is inferred from account name, account type, and description.",
        "The variance baseline is a trailing 3-month rolling average of monthly net cash flow.",
        "Variance is computed as actual minus the rolling average.",
        "Outliers are flagged when |variance| exceeds the larger of 20% of the baseline or 2,000.",
        "trial_balance is intentionally not required for this workbook; it is only a reconciliation reference when present.",
    ]
    for idx, item in enumerate(assumptions, start=2):
        logic_ws.write(idx, 0, f"- {item}", note_fmt)

    logic_ws.write(len(assumptions) + 4, 0, "Source bundle summary", section_fmt)
    summary_lines = [
        f"General ledger rows available: {len(gl_rows)}",
        f"Cash transactions processed: {len(processed_rows)}",
        f"Months analysed: {len(months)}",
        f"COA mapping entries: {coa_entries_written}",
        f"Trial balance present: {'yes' if isinstance(trial_balance, dict) else 'no'}",
    ]
    for idx, item in enumerate(summary_lines, start=len(assumptions) + 5):
        logic_ws.write(idx, 0, item, text_fmt)

    workbook.close()

    print(
        json.dumps(
            {
                "status": "ok",
                "bundle": str(bundle_path),
                "output": str(output_path),
                "gl_rows": len(gl_rows),
                "cash_rows": len(processed_rows),
                "months": len(months),
                "coa_entries": coa_entries_written,
                "total_net_cash_flow": total_cash_flow,
                "max_month": max_month,
                "min_month": min_month,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
