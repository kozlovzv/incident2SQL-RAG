from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from .database import init_db, Category, Incident
import logging
import random

logger = logging.getLogger(__name__)


def init_sample_data(db_url: str = "sqlite:///data/incidents.db"):
    """Initialize database with sample operational risk incidents"""
    engine = init_db(db_url)

    with Session(engine) as session:
        # Check if data already exists
        if session.query(Category).count() > 0:
            logger.info("Sample data already exists, skipping initialization")
            return engine

        logger.info("Initializing sample data...")

        # Create expanded categories
        categories = [
            Category(
                name="Network Outage",
                description="Complete or partial network service interruptions",
            ),
            Category(
                name="Server Crash", description="Server system failures and crashes"
            ),
            Category(
                name="Data Breach",
                description="Security breaches involving data exposure",
            ),
            Category(
                name="Hardware Failure",
                description="Physical hardware malfunctions and failures",
            ),
            Category(
                name="Security Incident",
                description="General security-related incidents",
            ),
            Category(
                name="System Performance",
                description="System performance degradation issues",
            ),
        ]
        session.add_all(categories)
        session.commit()

        # Create sample incidents with varied dates, risk levels, and statuses
        now = datetime.utcnow()
        statuses = ["new", "in_progress", "resolved", "closed"]
        risk_levels = [0.5, 0.7, 0.9, 1.0]  # Varied risk levels

        incidents = []
        # Last quarter incidents
        for i in range(30):  # Create 30 diverse incidents
            days_ago = random.randint(1, 90)  # Random date within last quarter
            incidents.append(
                Incident(
                    title=f"Incident {i + 1}",
                    description=generate_incident_description(i, categories),
                    date_occurred=now - timedelta(days=days_ago),
                    risk_level=random.choice(risk_levels),
                    status=random.choice(statuses),
                    category_id=random.randint(1, len(categories)),
                )
            )

        session.add_all(incidents)
        session.commit()
        logger.info(
            f"Created {len(categories)} categories and {len(incidents)} incidents"
        )

    return engine


def generate_incident_description(index: int, categories: list) -> str:
    """Generate meaningful description based on incident type"""
    descriptions = [
        "Critical network outage affecting all trading operations in the US region",
        "Multiple server crashes detected in the primary data center",
        "Potential data breach detected with unauthorized access attempts",
        "Hardware failure in the main storage array causing data access issues",
        "Security vulnerability exploited in the authentication system",
        "Severe performance degradation in the core banking system",
        "Network connectivity issues affecting remote offices",
        "Database server crash during peak hours",
        "Suspected data exfiltration from customer database",
        "Storage system hardware malfunction",
        "Multiple failed login attempts detected from suspicious IPs",
        "Critical system slowdown affecting user operations",
        "Fiber optic cable damage causing network disruption",
        "Application server cluster failure",
        "Unauthorized access to sensitive data detected",
        "RAID controller failure in primary storage",
        "Malware detection in corporate network",
        "API gateway performance degradation",
        "Network switch failure in backup data center",
        "Virtual machine host crash affecting multiple services",
        "Data corruption incident in backup systems",
        "Power supply failure in server rack",
        "DDoS attack attempt detected",
        "Database performance issues affecting transactions",
        "Network routing misconfiguration",
        "Cluster node failure in processing farm",
        "Unauthorized data access attempt blocked",
        "Storage array controller malfunction",
        "Firewall rule violation detected",
        "Load balancer performance degradation",
    ]
    return descriptions[index % len(descriptions)]


if __name__ == "__main__":
    init_sample_data()
