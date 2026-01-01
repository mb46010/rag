"""Configuration for the HR Assistant agent."""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the HR workflow agent."""

    # MCP settings
    mcp_url: str = "http://localhost:9000"

    # LLM settings
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048

    # Retry settings
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 10.0

    # Feature flags
    enable_streaming: bool = True
    enable_parallel_fetch: bool = True

    # Observability
    enable_langfuse: bool = True
