from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from statistics import mean, median, stdev
from datetime import datetime
import json


@dataclass
class QueryMetrics:
    query: str
    latency: float
    num_results: int
    avg_similarity: float
    cache_hits: int
    total_candidates: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class RAGMonitor:
    def __init__(self, log_file: Optional[str] = None):
        self.metrics: List[QueryMetrics] = []
        self.logger = logging.getLogger(__name__)

        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)
        self._setup_performance_tracking()
        self._setup_alert_thresholds()

    def _setup_performance_tracking(self):
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            "total_queries": 0,
            "cache_performance": {"hits": 0, "misses": 0},
            "latency_buckets": {
                "fast": 0,  # < 100ms
                "medium": 0,  # 100ms - 500ms
                "slow": 0,  # > 500ms
            },
            "hourly_stats": {},
            "error_count": 0,
            "quality_metrics": {
                "low_similarity_queries": 0,
                "empty_result_queries": 0,
                "perfect_match_queries": 0,
            },
        }

    def _setup_alert_thresholds(self):
        """Setup monitoring thresholds for alerts"""
        self.alert_thresholds = {
            "latency_ms": 1000,  # Alert if query takes more than 1 second
            "error_rate": 0.1,  # Alert if error rate exceeds 10%
            "empty_results_rate": 0.2,  # Alert if more than 20% queries return empty
            "low_similarity_threshold": 0.3,  # Alert if similarity below this
        }

    def log_query(
        self,
        query: str,
        results: List[Dict],
        duration: float,
        cache_hits: int = 0,
        total_candidates: int = 0,
        error: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Enhanced query logging with alerts and custom metrics"""
        try:
            # Calculate metrics
            avg_sim = mean([r["score"] for r in results]) if results else 0

            # Create metrics object with custom metrics
            metric = QueryMetrics(
                query=query,
                latency=duration,
                num_results=len(results),
                avg_similarity=avg_sim,
                cache_hits=cache_hits,
                total_candidates=total_candidates,
                custom_metrics=custom_metrics or {},
            )
            self.metrics.append(metric)

            # Update performance tracking
            self._update_performance_metrics(metric, error)

            # Check for alerts
            alerts = self._check_alerts(metric)

            # Log the query details
            log_entry = {
                "timestamp": metric.timestamp.isoformat(),
                "query": query,
                "results_count": len(results),
                "latency_ms": duration * 1000,
                "avg_similarity": avg_sim,
                "cache_hits": cache_hits,
                "total_candidates": total_candidates,
            }

            if custom_metrics:
                log_entry["custom_metrics"] = custom_metrics

            if alerts:
                log_entry["alerts"] = alerts
                self.logger.warning(f"Query alerts triggered: {json.dumps(log_entry)}")
            elif error:
                log_entry["error"] = error
                self.logger.error(f"Query error: {json.dumps(log_entry)}")
            else:
                self.logger.info(f"Query metrics: {json.dumps(log_entry)}")

        except Exception as e:
            self.logger.error(f"Error logging metrics: {str(e)}")

    def _check_alerts(self, metric: QueryMetrics) -> List[str]:
        """Check for alert conditions"""
        alerts = []

        # Check latency
        if metric.latency * 1000 > self.alert_thresholds["latency_ms"]:
            alerts.append(f"High latency: {metric.latency * 1000:.2f}ms")

        # Check similarity
        if (
            metric.num_results > 0
            and metric.avg_similarity
            < self.alert_thresholds["low_similarity_threshold"]
        ):
            alerts.append(f"Low similarity score: {metric.avg_similarity:.2f}")

        # Check if query returned no results
        if metric.num_results == 0:
            alerts.append("Query returned no results")

        # Calculate current error rate
        if len(self.metrics) >= 100:  # Use rolling window of last 100 queries
            recent_errors = sum(
                1 for m in self.metrics[-100:] if "error" in m.custom_metrics
            )
            error_rate = recent_errors / 100
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append(f"High error rate: {error_rate:.2%}")

        return alerts

    def _update_performance_metrics(
        self, metric: QueryMetrics, error: Optional[str] = None
    ):
        """Update internal performance tracking metrics"""
        self.performance_metrics["total_queries"] += 1

        # Update quality metrics
        if metric.num_results == 0:
            self.performance_metrics["quality_metrics"]["empty_result_queries"] += 1
        elif metric.avg_similarity < self.alert_thresholds["low_similarity_threshold"]:
            self.performance_metrics["quality_metrics"]["low_similarity_queries"] += 1
        elif metric.avg_similarity > 0.9:  # Consider as perfect match
            self.performance_metrics["quality_metrics"]["perfect_match_queries"] += 1

        # Update latency buckets
        latency_ms = metric.latency * 1000
        if latency_ms < 100:
            self.performance_metrics["latency_buckets"]["fast"] += 1
        elif latency_ms < 500:
            self.performance_metrics["latency_buckets"]["medium"] += 1
        else:
            self.performance_metrics["latency_buckets"]["slow"] += 1

        # Update cache performance
        if metric.cache_hits > 0:
            self.performance_metrics["cache_performance"]["hits"] += metric.cache_hits
        self.performance_metrics["cache_performance"]["misses"] += (
            metric.total_candidates - metric.cache_hits
        )

        # Update hourly stats
        hour = metric.timestamp.strftime("%Y-%m-%d %H:00")
        if hour not in self.performance_metrics["hourly_stats"]:
            self.performance_metrics["hourly_stats"][hour] = {
                "query_count": 0,
                "avg_latency": 0,
                "total_results": 0,
                "cache_hit_rate": 0,
            }

        hour_stats = self.performance_metrics["hourly_stats"][hour]
        hour_stats["query_count"] += 1
        hour_stats["avg_latency"] = (
            hour_stats["avg_latency"] * (hour_stats["query_count"] - 1) + metric.latency
        ) / hour_stats["query_count"]
        hour_stats["total_results"] += metric.num_results
        hour_stats["cache_hit_rate"] = (
            self.performance_metrics["cache_performance"]["hits"]
            / (
                self.performance_metrics["cache_performance"]["hits"]
                + self.performance_metrics["cache_performance"]["misses"]
            )
            if self.performance_metrics["cache_performance"]["hits"] > 0
            else 0
        )

        if error:
            self.performance_metrics["error_count"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.metrics:
            return {}

        recent_queries = self.metrics[-100:]  # Last 100 queries for recent stats

        try:
            latencies = [m.latency for m in recent_queries]
            similarities = [
                m.avg_similarity for m in recent_queries if m.avg_similarity > 0
            ]

            quality_metrics = self.performance_metrics["quality_metrics"]
            total_queries = len(self.metrics)

            stats = {
                # Overall stats
                "total_queries": total_queries,
                "recent_queries": len(recent_queries),
                # Latency stats (keep old format for compatibility)
                "avg_latency": mean(latencies) if latencies else 0,
                "latency_details": {
                    "avg_ms": mean(latencies) * 1000 if latencies else 0,
                    "median_ms": median(latencies) * 1000 if latencies else 0,
                    "std_dev_ms": stdev(latencies) * 1000 if len(latencies) > 1 else 0,
                    "percentiles": {
                        "p95": sorted(latencies)[int(len(latencies) * 0.95)] * 1000
                        if latencies
                        else 0,
                        "p99": sorted(latencies)[int(len(latencies) * 0.99)] * 1000
                        if latencies
                        else 0,
                    }
                    if latencies
                    else {},
                },
                # Result quality stats
                "quality": {
                    "empty_result_rate": quality_metrics["empty_result_queries"]
                    / total_queries,
                    "low_similarity_rate": quality_metrics["low_similarity_queries"]
                    / total_queries,
                    "perfect_match_rate": quality_metrics["perfect_match_queries"]
                    / total_queries,
                    "avg_similarity": mean(similarities) if similarities else 0,
                },
                # Cache performance
                "cache_performance": {
                    "hit_rate": self.performance_metrics["cache_performance"]["hits"]
                    / (
                        self.performance_metrics["cache_performance"]["hits"]
                        + self.performance_metrics["cache_performance"]["misses"]
                    )
                    if self.performance_metrics["cache_performance"]["hits"] > 0
                    else 0
                },
                # Error stats
                "error_rate": self.performance_metrics["error_count"] / total_queries,
                # Performance distribution
                "latency_distribution": {
                    k: v / total_queries
                    for k, v in self.performance_metrics["latency_buckets"].items()
                },
                # Hourly trends
                "hourly_trends": self.performance_metrics["hourly_stats"],
            }

            return stats

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            return {"total_queries": len(self.metrics), "error": str(e)}
