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

        self.date_patterns = {
            "count": ["количество", "число", "сколько", "count", "number of"],
            "grouping": ["по дням", "за день", "в день", "per day", "by day"],
            "max": ["больше всего", "максимальное", "наибольшее", "maximum", "most"],
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

        try:
            query_lower = query.lower()

            # Check if this is a request for all records without limit
            no_limit = any(
                phrase in query_lower
                for phrase in [
                    "без ограничений",
                    "все инциденты",
                    "without limit",
                    "all incidents",
                ]
            )

            # Check if this is an aggregation query
            is_count_query = any(
                word in query_lower for word in self.date_patterns["count"]
            )
            is_group_by_day = any(
                word in query_lower for word in self.date_patterns["grouping"]
            )
            is_max_count = any(
                word in query_lower for word in self.date_patterns["max"]
            )

            if is_count_query or is_group_by_day or is_max_count:
                # Build aggregation query
                base_sql = """
                    WITH daily_counts AS (
                        SELECT 
                            date(date_occurred) as incident_date,
                            COUNT(*) as incident_count
                        FROM incidents i
                        GROUP BY date(date_occurred)
                    )
                """

                if is_max_count:
                    base_sql += """
                        SELECT incident_date, incident_count
                        FROM daily_counts
                        ORDER BY incident_count DESC, incident_date DESC
                        LIMIT 1
                    """
                else:
                    base_sql += """
                        SELECT incident_date, incident_count
                        FROM daily_counts
                        ORDER BY incident_date DESC
                    """

                return base_sql.strip()

            # Regular query processing
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

            # Time period processing
            time_condition = self._process_time_period(query)
            if time_condition:
                conditions.append(time_condition)

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

            # Apply limit only if not explicitly requested to show all
            if not no_limit:
                # Limit (with fallback)
                limit = 10
                try:
                    limit_match = re.search(
                        r"(\d+).*?(?:инцидент|incident)", query_lower
                    )
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
        query_lower = query.lower()

        # Time period keywords
        month_keywords = ["last month", "последний месяц", "прошлый месяц", "за месяц"]
        week_keywords = [
            "last week",
            "последняя неделя",
            "прошлая неделя",
            "за неделю",
            "последнюю неделю",
        ]

        # Check for specific date (format: DD month YYYY)
        date_pattern = r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})"
        date_match = re.search(date_pattern, query_lower)
        if date_match:
            month_mapping = {
                "января": "01",
                "февраля": "02",
                "марта": "03",
                "апреля": "04",
                "мая": "05",
                "июня": "06",
                "июля": "07",
                "августа": "08",
                "сентября": "09",
                "октября": "10",
                "ноября": "11",
                "декабря": "12",
            }
            day = date_match.group(1).zfill(2)
            month = month_mapping[date_match.group(2)]
            year = date_match.group(3)
            return f"date(i.date_occurred) = '{year}-{month}-{day}'"

        # Check for week period
        if any(keyword in query_lower for keyword in week_keywords):
            return "i.date_occurred >= date('now', '-7 days')"

        # Check for month period
        if any(keyword in query_lower for keyword in month_keywords):
            return "i.date_occurred >= date('now', '-1 month')"

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
