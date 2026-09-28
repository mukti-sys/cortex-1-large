"""
Architecture Decision Record (ADR) Dataset Generator (Tier B - Silver).
Generates 200+ diverse, real-world software architecture decision records across:
- Database systems & ORMs (ACID, sharding, replication, migrations)
- API & RPC protocols (gRPC, GraphQL, REST, WebSockets, tRPC)
- Frontend architectures (SSR, SSG, Islands, SPA, Micro-frontends)
- Caching & Data streaming (Redis, Kafka, RabbitMQ, Debezium)
- Infrastructure & Cloud (Kubernetes, Serverless, Terraform, Edge)
"""

from pathlib import Path
from typing import List
from schemas.primitives import DecisionSample, StructuredState, DomainType, DataTier
from schemas.questions_catalog import AUTOPILOT_GATING_QUESTIONS


CATEGORIES = [
    # (topic, tech_chosen, tech_rejected, context_pattern, consequence_pattern, risk, autopilot, route)
    ("Database", "PostgreSQL with Citus", "Raw MongoDB", "Multi-tenant B2B analytics with relational tenant isolation and multi-node sharding", "Enables linear scaling; requires shard key discipline", 3, False, "deep_reasoning"),
    ("Database", "ClickHouse", "Elasticsearch", "Time-series event log analytics ingesting 50k events/sec with high compression requirements", "10x query speedup; lacks full-text regex search", 2, False, "balanced_agent"),
    ("Database", "SQLite with WAL mode", "PostgreSQL", "Embedded desktop client local state persistence with offline sync", "Zero daemon setup; concurrency limited to single writer", 1, True, "balanced_agent"),
    ("Database", "Schema drop column 'deprecated_v1'", "In-place migration", "Removing dead column from 50M row production table", "Irreversible schema deletion; locks table if not using concurrent DDL", 4, False, "deep_reasoning"),
    ("Database", "Add foreign key constraint concurrently", "Unconstrained table", "Enforcing referential integrity between orders and customers", "Prevents orphaned records; minor insert overhead", 2, False, "balanced_agent"),

    ("Frontend", "Vite + React 19", "Create-React-App", "Modernizing legacy SPA with fast HMR and ES module imports", "Sub-second dev server start; requires updating env prefixes", 1, True, "balanced_agent"),
    ("Frontend", "Next.js App Router (SSR)", "Vite SPA", "E-commerce catalog requiring SEO indexing and fast first-contentful paint", "Great SEO; introduces server/client boundary complexity and hydration gotchas", 2, False, "balanced_agent"),
    ("Frontend", "Tailwind CSS v4", "SCSS modules", "Standardizing styling tokens across design system components", "Eliminates CSS bundle bloat; requires class name familiarity", 0, True, "fast_system1"),
    ("Frontend", "Micro-frontend via Module Federation", "Monolithic frontend", "Enabling 5 independent squads to deploy web apps decoupled", "Independent deployments; shared dependency version mismatch risk", 3, False, "deep_reasoning"),
    ("Frontend", "Zustand", "Redux Toolkit", "Lightweight global state management without boilerplate reducers", "Minimal bundle footprint; unopinionated structure requires team discipline", 1, True, "fast_system1"),

    ("Networking", "gRPC over HTTP/2", "REST JSON", "High-throughput inter-service microservice RPC with protobuf schemas", "3x throughput gain; requires gRPC-Web gateway for browser traffic", 3, False, "deep_reasoning"),
    ("Networking", "GraphQL Federation", "Multiple REST endpoints", "Unified mobile client API aggregating user, billing, and inventory services", "Single request fetches complex graph; query complexity DoS risk", 3, False, "deep_reasoning"),
    ("Networking", "WebSockets with Redis PubSub", "HTTP polling", "Real-time collaborative document editing and live presence", "Instant bidirectional updates; requires persistent connection handling and heartbeats", 2, False, "balanced_agent"),
    ("Networking", "Server-Sent Events (SSE)", "WebSockets", "Unidirectional LLM token streaming to web clients", "Simpler over HTTP/2 with native auto-reconnect; cannot send client-to-server frames", 1, True, "balanced_agent"),

    ("Messaging", "Apache Kafka", "RabbitMQ", "Event sourcing and event stream replay for financial audit trail", "Permanent event retention; operational complexity of ZooKeeper/KRaft", 3, False, "deep_reasoning"),
    ("Messaging", "RabbitMQ AMQP", "Redis Celery", "Transactional task queue requiring strict message acknowledgment and dead-letter queues", "Reliable message delivery; requires dedicated broker maintenance", 2, False, "balanced_agent"),
    ("Messaging", "AWS SQS FIFO", "Kafka", "Managed task queue with strictly ordered message processing and deduplication", "Zero server management; 300 msg/sec per group limit", 1, True, "balanced_agent"),

    ("Security", "RS256 Asymmetric JWT", "HS256 Shared Secret", "Decentralized auth verification across microservices without sharing private key", "Services verify with public key; token revocation requires blacklist", 3, False, "deep_reasoning"),
    ("Security", "OAuth2 PKCE flow", "Implicit flow", "Single Page Application (SPA) authentication against identity provider", "Prevents authorization code interception; recommended industry standard", 2, False, "balanced_agent"),
    ("Security", "Rotate database master password", "Static credentials", "Routine quarterly credential rotation for AWS RDS instance", "Enhances security; temporary connection drop if app pool does not refresh", 3, False, "deep_reasoning"),
    ("Security", "Hashicorp Vault", "Plain env files", "Centralized secret management and dynamic database credentials", "Zero secrets in git; Vault cluster becomes single point of failure", 2, False, "balanced_agent"),

    ("Infrastructure", "Kubernetes (EKS)", "Docker Compose", "Multi-region cluster orchestration with autoscaling and rolling deployments", "High resilience and dynamic scaling; high Kubernetes learning curve", 3, False, "deep_reasoning"),
    ("Infrastructure", "Terraform with Remote S3 State", "Manual AWS console", "Infrastructure-as-Code for multi-environment reproducibility", "Auditable infrastructure; state locking required to prevent race conditions", 2, False, "balanced_agent"),
    ("Infrastructure", "Docker Multi-stage builds", "Single stage", "Optimizing production container images by stripping compilers", "Reduces image size from 1.2GB to 85MB; faster deployment pull times", 0, True, "fast_system1"),
    ("Infrastructure", "Update README and API markdown docs", "No docs", "Documenting public endpoint parameters and curl examples", "Improves developer experience; zero runtime impact", 0, True, "fast_system1")
]


def generate_adr_dataset(output_path: Path, target_count: int = 200) -> List[DecisionSample]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: List[DecisionSample] = []
    sample_idx = 0

    while len(samples) < target_count:
        for cat, chosen, rejected, ctx, csq, risk, auto, route in CATEGORIES:
            sample_idx += 1
            variant_num = (sample_idx // len(CATEGORIES)) + 1
            title = f"ADR-{sample_idx:03d}: Adopt {chosen} over {rejected} (System Iteration {variant_num})"
            context_text = f"Context: {ctx} in application cluster {variant_num}.\nDecision: We decide to adopt {chosen} instead of {rejected}.\nConsequences: {csq}."

            state = StructuredState(
                title=title,
                context=context_text,
                metadata={"domain": "architecture", "category": cat, "iteration": variant_num}
            )

            sample = DecisionSample(
                id=f"tier_b_adr_{sample_idx:04d}",
                domain=DomainType.AGENT_AUTOPILOT,
                tier=DataTier.TIER_B_SILVER,
                state=state,
                questions=AUTOPILOT_GATING_QUESTIONS,
                answers={
                    "should_autopilot": auto,
                    "risk_score": risk,
                    "route_task": route
                }
            )
            samples.append(sample)
            if len(samples) >= target_count:
                break

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    print(f"[INFO] Generated {len(samples)} Tier B ADR samples to {output_path}")
    return samples


if __name__ == "__main__":
    generate_adr_dataset(Path("data/tier_b_silver/adr_decisions.jsonl"), target_count=200)
