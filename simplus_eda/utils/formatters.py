"""
Output formatting utilities.

Provides comprehensive formatting for analysis results including:
- JSON formatting (pretty and compact)
- Markdown table and text formatting
- HTML table formatting
- CSV export
- Number, percentage, and currency formatting
- File size and duration formatting
- Statistical summary formatting
"""

from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd
from datetime import datetime, timedelta


class OutputFormatter:
    """
    Format analysis results for various output formats.

    Supports formatting to:
    - JSON (pretty-printed or compact)
    - Markdown (tables, headers, lists)
    - HTML (tables, formatted text)
    - CSV (for tabular data)
    - Plain text (human-readable summaries)

    Also provides number formatting utilities:
    - Percentages
    - Large numbers with suffixes (K, M, B)
    - Currency
    - File sizes
    - Durations
    """

    @staticmethod
    def to_json(data: Dict[str, Any], pretty: bool = True, sort_keys: bool = False) -> str:
        """
        Convert results to JSON string.

        Args:
            data: Dictionary to convert
            pretty: Whether to pretty-print the JSON
            sort_keys: Whether to sort dictionary keys

        Returns:
            JSON string
        """
        if pretty:
            return json.dumps(data, indent=2, default=str, sort_keys=sort_keys)
        return json.dumps(data, default=str, sort_keys=sort_keys)

    @staticmethod
    def to_markdown(data: Dict[str, Any], title: Optional[str] = None) -> str:
        """
        Convert results to markdown format.

        Args:
            data: Dictionary to convert
            title: Optional title for the markdown document

        Returns:
            Markdown formatted string
        """
        lines = []

        if title:
            lines.append(f"# {title}\n")

        lines.append(OutputFormatter._dict_to_markdown(data))

        return "\n".join(lines)

    @staticmethod
    def _dict_to_markdown(data: Dict[str, Any], level: int = 2) -> str:
        """
        Recursively convert dictionary to markdown.

        Args:
            data: Dictionary to convert
            level: Header level for nested dictionaries

        Returns:
            Markdown formatted string
        """
        lines = []

        for key, value in data.items():
            # Format key as header
            header_prefix = "#" * level
            lines.append(f"{header_prefix} {key.replace('_', ' ').title()}\n")

            if isinstance(value, dict):
                # Recursively format nested dictionaries
                if level < 6:  # Limit header depth
                    lines.append(OutputFormatter._dict_to_markdown(value, level + 1))
                else:
                    # Format as list if too deep
                    for k, v in value.items():
                        lines.append(f"- **{k}**: {v}")
                    lines.append("")

            elif isinstance(value, (list, tuple)):
                # Format as unordered list
                for item in value:
                    if isinstance(item, dict):
                        lines.append("- " + ", ".join(f"{k}: {v}" for k, v in item.items()))
                    else:
                        lines.append(f"- {item}")
                lines.append("")

            elif isinstance(value, pd.DataFrame):
                # Format DataFrame as markdown table
                lines.append(value.to_markdown(index=False))
                lines.append("")

            else:
                # Format simple values
                lines.append(f"{value}\n")

        return "\n".join(lines)

    @staticmethod
    def to_html_table(data: Union[pd.DataFrame, List[Dict[str, Any]]],
                     title: Optional[str] = None,
                     classes: str = "table") -> str:
        """
        Convert data to HTML table.

        Args:
            data: DataFrame or list of dictionaries
            title: Optional table title
            classes: CSS classes for the table

        Returns:
            HTML table string
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)

        html_parts = []

        if title:
            html_parts.append(f"<h3>{title}</h3>")

        html_parts.append(data.to_html(classes=classes, index=False))

        return "\n".join(html_parts)

    @staticmethod
    def to_csv(data: Union[pd.DataFrame, List[Dict[str, Any]]],
               include_index: bool = False) -> str:
        """
        Convert data to CSV string.

        Args:
            data: DataFrame or list of dictionaries
            include_index: Whether to include index in output

        Returns:
            CSV formatted string
        """
        if isinstance(data, list):
            data = pd.DataFrame(data)

        return data.to_csv(index=include_index)

    @staticmethod
    def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
        """
        Format value as percentage.

        Args:
            value: Numeric value (0-1 or 0-100)
            decimals: Number of decimal places
            include_sign: Whether to include % sign

        Returns:
            Formatted percentage string
        """
        if value <= 1:
            value *= 100

        formatted = f"{value:.{decimals}f}"

        if include_sign:
            formatted += "%"

        return formatted

    @staticmethod
    def format_number(value: float,
                     decimals: int = 2,
                     thousands_separator: bool = True) -> str:
        """
        Format number with specified decimals and optional thousands separator.

        Args:
            value: Numeric value
            decimals: Number of decimal places
            thousands_separator: Whether to include thousands separator

        Returns:
            Formatted number string
        """
        if thousands_separator:
            return f"{value:,.{decimals}f}"
        return f"{value:.{decimals}f}"

    @staticmethod
    def format_large_number(value: float, decimals: int = 1) -> str:
        """
        Format large numbers with K/M/B/T suffixes.

        Args:
            value: Numeric value
            decimals: Number of decimal places

        Returns:
            Formatted number string with suffix
        """
        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if abs_value >= 1_000_000_000_000:
            return f"{sign}{abs_value / 1_000_000_000_000:.{decimals}f}T"
        elif abs_value >= 1_000_000_000:
            return f"{sign}{abs_value / 1_000_000_000:.{decimals}f}B"
        elif abs_value >= 1_000_000:
            return f"{sign}{abs_value / 1_000_000:.{decimals}f}M"
        elif abs_value >= 1_000:
            return f"{sign}{abs_value / 1_000:.{decimals}f}K"
        else:
            return f"{sign}{abs_value:.{decimals}f}"

    @staticmethod
    def format_currency(value: float,
                       currency_symbol: str = "$",
                       decimals: int = 2) -> str:
        """
        Format value as currency.

        Args:
            value: Numeric value
            currency_symbol: Currency symbol to use
            decimals: Number of decimal places

        Returns:
            Formatted currency string
        """
        formatted = OutputFormatter.format_number(value, decimals)
        if value < 0:
            # Move minus sign before currency symbol
            formatted = formatted.lstrip("-")
            return f"-{currency_symbol}{formatted}"
        return f"{currency_symbol}{formatted}"

    @staticmethod
    def format_file_size(bytes_value: int, decimals: int = 2) -> str:
        """
        Format file size in human-readable format.

        Args:
            bytes_value: Size in bytes
            decimals: Number of decimal places

        Returns:
            Formatted file size string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if abs(bytes_value) < 1024.0:
                return f"{bytes_value:.{decimals}f} {unit}"
            bytes_value /= 1024.0

        return f"{bytes_value:.{decimals}f} EB"

    @staticmethod
    def format_duration(seconds: float,
                       verbose: bool = False,
                       max_units: int = 2) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds
            verbose: Use verbose format (e.g., "1 hour, 30 minutes" vs "1h 30m")
            max_units: Maximum number of time units to show

        Returns:
            Formatted duration string
        """
        if seconds == 0:
            return "0s" if not verbose else "0 seconds"

        delta = timedelta(seconds=seconds)
        days = delta.days
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        parts = []

        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}" if verbose else f"{days}d")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}" if verbose else f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}" if verbose else f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}" if verbose else f"{secs}s")

        # Limit to max_units
        parts = parts[:max_units]

        return ", ".join(parts) if verbose else " ".join(parts)

    @staticmethod
    def format_timestamp(dt: datetime,
                        format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format datetime as string.

        Args:
            dt: Datetime object
            format_string: Format string for strftime

        Returns:
            Formatted datetime string
        """
        return dt.strftime(format_string)

    @staticmethod
    def format_statistical_summary(data: Dict[str, Any],
                                   decimals: int = 2) -> str:
        """
        Format statistical summary in human-readable text.

        Args:
            data: Dictionary containing statistical data
            decimals: Number of decimal places for numbers

        Returns:
            Formatted summary string
        """
        lines = []

        lines.append("Statistical Summary")
        lines.append("=" * 50)

        for key, value in data.items():
            formatted_key = key.replace("_", " ").title()

            if isinstance(value, dict):
                lines.append(f"\n{formatted_key}:")
                for sub_key, sub_value in value.items():
                    formatted_sub_key = sub_key.replace("_", " ").title()
                    formatted_value = OutputFormatter._format_value(sub_value, decimals)
                    lines.append(f"  {formatted_sub_key}: {formatted_value}")

            elif isinstance(value, (list, tuple)):
                lines.append(f"\n{formatted_key}:")
                for item in value:
                    lines.append(f"  - {item}")

            else:
                formatted_value = OutputFormatter._format_value(value, decimals)
                lines.append(f"{formatted_key}: {formatted_value}")

        return "\n".join(lines)

    @staticmethod
    def _format_value(value: Any, decimals: int = 2) -> str:
        """
        Format a single value appropriately based on its type.

        Args:
            value: Value to format
            decimals: Number of decimal places for floats

        Returns:
            Formatted value string
        """
        if isinstance(value, float):
            # Check if it's a percentage-like value (0-1)
            if 0 <= value <= 1 and value != int(value):
                return OutputFormatter.format_percentage(value, decimals)
            else:
                return OutputFormatter.format_number(value, decimals)

        elif isinstance(value, int):
            if value > 1_000_000:
                return OutputFormatter.format_large_number(value, decimals)
            else:
                return OutputFormatter.format_number(value, 0)

        elif isinstance(value, datetime):
            return OutputFormatter.format_timestamp(value)

        else:
            return str(value)

    @staticmethod
    def create_markdown_table(headers: List[str],
                             rows: List[List[Any]],
                             align: Optional[List[str]] = None) -> str:
        """
        Create a markdown table.

        Args:
            headers: List of column headers
            rows: List of row data (list of lists)
            align: Optional list of alignments ('left', 'center', 'right')

        Returns:
            Markdown table string
        """
        if not headers or not rows:
            return ""

        # Determine column alignments
        if align is None:
            align = ['left'] * len(headers)

        # Create alignment indicators
        align_chars = {
            'left': ':--',
            'center': ':-:',
            'right': '--:'
        }

        # Build header row
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"

        # Build separator row
        separator_row = "| " + " | ".join(
            align_chars.get(a, ':--') for a in align
        ) + " |"

        # Build data rows
        data_rows = []
        for row in rows:
            # Ensure row has same length as headers
            row_data = [str(item) if item is not None else "" for item in row]
            while len(row_data) < len(headers):
                row_data.append("")
            data_rows.append("| " + " | ".join(row_data[:len(headers)]) + " |")

        # Combine all parts
        table_lines = [header_row, separator_row] + data_rows

        return "\n".join(table_lines)

    @staticmethod
    def create_comparison_table(before: Dict[str, Any],
                               after: Dict[str, Any],
                               title: str = "Comparison") -> str:
        """
        Create a comparison table in markdown format.

        Args:
            before: Dictionary of before values
            after: Dictionary of after values
            title: Table title

        Returns:
            Markdown comparison table
        """
        headers = ["Metric", "Before", "After", "Change"]
        rows = []

        all_keys = set(before.keys()) | set(after.keys())

        for key in sorted(all_keys):
            before_val = before.get(key, "N/A")
            after_val = after.get(key, "N/A")

            # Calculate change if both are numeric
            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                change = after_val - before_val
                change_str = f"{change:+.2f}" if isinstance(change, float) else f"{change:+d}"
            else:
                change_str = "-"

            rows.append([
                key.replace("_", " ").title(),
                str(before_val),
                str(after_val),
                change_str
            ])

        table = OutputFormatter.create_markdown_table(headers, rows)

        return f"## {title}\n\n{table}"
