from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Optional
import re


class Text2SQLGenerator:
    def __init__(self, model_name: str = "juierror/text-to-sql-with-table-schema"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        # Russian language patterns
        self.risk_patterns = {
            "high": ["высок", "критическ", "критичн"],
            "medium": ["средн"],
            "low": ["низк", "минимальн"],
            "comparison": {
                "greater": ["выше", "больше", "более"],
                "less": ["ниже", "меньше", "менее"],
            },
        }

    def _clean_sql(self, sql: str) -> str:
        """Validate and clean SQL query"""
        # Basic SQL injection prevention
        dangerous_patterns = [
            r";\s*\w+",  # Multiple statements
            r"--.*$",  # Line comments
            r"/\*.*?\*/",  # Block comments
            r"\bUNION\b.*\bSELECT\b",
            r"\bDROP\b.*\bTABLE\b",
            r"\bDELETE\b.*\bFROM\b",
            r"\bUPDATE\b.*\bSET\b",
            r"\bINSERT\b.*\bINTO\b",
            r"\bALTER\b.*\bTABLE\b",
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE):
                raise ValueError("SQL injection attempt detected")

        return sql.strip()

    def generate_sql(self, query: str, context: Optional[List[Dict]] = None) -> str:
        if not query:
            raise ValueError("Query cannot be empty")

        # SQL injection check
        if any(char in query for char in [";", "--", "/*", "*/"]):
            raise ValueError("SQL injection attempt detected")

        try:
            base_sql = "SELECT i.*, c.name as category_name, c.description as category_description "
            base_sql += "FROM incidents i LEFT JOIN categories c ON i.category_id = c.id WHERE 1=1"

            conditions = []
            sort_criteria = []
            query_lower = query.lower()

            # Status processing
            status_keywords = ["status", "со статусом", "статус"]
            if any(keyword in query_lower for keyword in status_keywords):
                for word in ["resolved", "new", "in_progress", "closed"]:
                    if word in query_lower:
                        conditions.append(f"i.status = '{word}'")
                        break

            # Quarter processing
            quarter_pattern = r"q(\d)\s*(\d{4})"
            quarter_match = re.search(quarter_pattern, query_lower)
            if quarter_match:
                quarter = int(quarter_match.group(1))
                year = int(quarter_match.group(2))
                start_month = (quarter - 1) * 3 + 1
                end_month = start_month + 3
                conditions.append(
                    f"date_occurred >= '{year}-{start_month:02d}-01' AND "
                    f"date_occurred < '{year}-{end_month:02d}-01'"
                )

            # Risk level processing
            risk_pattern = r"(?:риск|risk).*?(0\.\d+)"
            risk_match = re.search(risk_pattern, query_lower)
            if risk_match:
                risk_value = risk_match.group(1)
                if any(
                    word in query_lower
                    for word in ["выше", "больше", "более", "higher", "greater"]
                ):
                    conditions.append(f"i.risk_level >= {risk_value}")
                elif any(
                    word in query_lower
                    for word in ["ниже", "меньше", "менее", "lower", "less"]
                ):
                    conditions.append(f"i.risk_level <= {risk_value}")
                else:
                    conditions.append(f"i.risk_level = {risk_value}")
            elif any(
                word in query_lower.split() for word in ["высок", "high", "critical"]
            ):
                conditions.append("i.risk_level >= 0.8")

            # Category processing
            if "server crash" in query_lower:
                conditions.append("c.name = 'Server Crash'")

            # Time period processing (non-quarter)
            if "last month" in query_lower or "последний месяц" in query_lower:
                conditions.append("i.date_occurred >= date('now', '-1 month')")
            elif "last week" in query_lower or "последняя неделя" in query_lower:
                conditions.append("i.date_occurred >= date('now', '-7 days')")

            # Sorting
            sort_type = None
            sort_order = None

            # Determine sort type
            if any(
                word in query_lower
                for word in ["дате", "date", "newest", "oldest", "старейшие", "новые"]
            ):
                sort_type = "date"
            elif any(word in query_lower for word in ["риск", "risk"]):
                sort_type = "risk"

            # Determine sort order
            if any(
                word in query_lower
                for word in [
                    "oldest",
                    "старейшие",
                    "сначала старые",
                    "ascending",
                    "возрастанию",
                ]
            ):
                sort_order = "ASC"
            elif any(
                word in query_lower
                for word in [
                    "newest",
                    "новые",
                    "сначала новые",
                    "descending",
                    "убыванию",
                ]
            ):
                sort_order = "DESC"
            else:
                sort_order = "DESC"  # default order

            # Apply sorting
            if sort_type == "date":
                sort_criteria.append(f"i.date_occurred {sort_order}")
            elif sort_type == "risk":
                sort_criteria.append(f"i.risk_level {sort_order}")

            # Build query
            if conditions:
                base_sql += " AND " + " AND ".join(conditions)

            # Primary sort criteria if specified, otherwise risk sort
            if sort_criteria:
                base_sql += " ORDER BY " + ", ".join(sort_criteria)
            else:
                base_sql += " ORDER BY i.risk_level DESC"

            # Limit (with fallback)
            limit = 10
            try:
                limit_match = re.search(r"(\d+).*?(?:инцидент|incident)", query_lower)
                if limit_match:
                    limit = int(limit_match.group(1))
            except (AttributeError, ValueError):
                pass

            base_sql += f" LIMIT {limit}"

            return base_sql.strip()
        except Exception as e:
            raise ValueError(f"Error generating SQL query: {str(e)}")

    def _extract_risk_level(self, query: str) -> Optional[str]:
        """Extract risk level from query"""
        # Поиск числового значения риска
        risk_levels = re.findall(r"0\.\d+", query)
        if risk_levels:
            return risk_levels[0]
        return None

    def _get_comparison_operator(self, query: str) -> str:
        """Get comparison operator based on query context"""
        for op_type, words in self.risk_patterns["comparison"].items():
            if any(word in query for word in words):
                return ">=" if op_type == "greater" else "<="
        return "="

    def _process_time_period(self, query: str) -> Optional[str]:
        """Process time period conditions"""
        if "last month" in query or "последний месяц" in query:
            return "i.date_occurred >= date('now', '-1 month')"
        elif "last week" in query or "последняя неделя" in query:
            return "i.date_occurred >= date('now', '-7 days')"
        return None

    def _process_sorting(self, query: str) -> List[str]:
        """Process sorting conditions"""
        criteria = []

        # Date sorting
        if any(word in query for word in ["старейшие", "oldest", "сначала старые"]):
            criteria.append("i.date_occurred ASC")
        elif any(word in query for word in ["новые", "newest", "сначала новые"]):
            criteria.append("i.date_occurred DESC")

        # Risk sorting
        if "риск" in query:
            order = "DESC" if "убыванию" in query else "ASC"
            criteria.append(f"i.risk_level {order}")

        return criteria

    def _extract_limit(self, query: str) -> int:
        """Extract limit from query"""
        limit_match = re.search(r"(\d+)\s+(?:инцидент|incident)", query)
        return int(limit_match.group(1)) if limit_match else 10

    def _has_sql_injection(self, query: str) -> bool:
        """Check for potential SQL injection"""
        dangerous_patterns = [
            ";",
            "--",
            "/*",
            "*/",
            "union",
            "drop",
            "delete",
            "update",
            "insert",
            "alter",
            "truncate",
        ]
        return any(pattern in query.lower() for pattern in dangerous_patterns)
