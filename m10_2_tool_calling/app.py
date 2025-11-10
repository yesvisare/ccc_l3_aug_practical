"""
FastAPI Application for Module 10.2: Tool Calling & Function Execution

Provides REST API endpoints for tool execution and agent queries.
No business logic - delegates to l2_m10_tool_calling_function_execution.py
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

# Import core module
from src.l3_m10_tool_calling_function_execution import (
    tool_registry,
    register_default_tools,
    SafeToolExecutor,
    ReActAgent,
    ToolResult
)
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Module 10.2: Tool Calling & Function Execution",
    description="Production-grade tool ecosystem for agentic RAG systems",
    version="1.0.0"
)

# Initialize tool system
register_default_tools(tool_registry)
executor = SafeToolExecutor(tool_registry)
agent = ReActAgent(executor)

logger.info("FastAPI app initialized with tool registry")


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    tools_registered: int
    services: Dict[str, bool]


class ToolExecutionRequest(BaseModel):
    """Request to execute a single tool."""
    tool_name: str = Field(..., description="Name of tool to execute")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")


class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float
    retries_used: int = 0
    skipped: bool = False
    reason: Optional[str] = None


class AgentQueryRequest(BaseModel):
    """Request for ReAct agent to answer query."""
    query: str = Field(..., description="User query for agent")
    max_iterations: Optional[int] = Field(10, description="Max reasoning iterations")


class AgentQueryResponse(BaseModel):
    """Response from agent query."""
    answer: str
    success: bool
    iterations: int
    trace: List[Dict[str, Any]]
    skipped: bool = False
    reason: Optional[str] = None


class ToolListResponse(BaseModel):
    """List of available tools."""
    tools: List[Dict[str, Any]]
    count: int


class StatsResponse(BaseModel):
    """Tool execution statistics."""
    stats: Dict[str, Dict[str, Any]]


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        System health status and configuration
    """
    services = config.validate_config()

    return HealthResponse(
        status="ok",
        tools_registered=len(tool_registry.list_tools()),
        services=services
    )


@app.get("/tools", response_model=ToolListResponse)
async def list_tools():
    """
    List all registered tools.

    Returns:
        List of tool definitions with schemas
    """
    tools = tool_registry.list_tools()
    tools_data = [
        {
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "parameters": tool.parameters,
            "timeout_seconds": tool.timeout_seconds,
            "retry_count": tool.retry_count
        }
        for tool in tools
    ]

    return ToolListResponse(
        tools=tools_data,
        count=len(tools_data)
    )


@app.post("/tool/execute", response_model=ToolExecutionResponse)
async def execute_tool(request: ToolExecutionRequest):
    """
    Execute a single tool.

    Args:
        request: Tool name and arguments

    Returns:
        Tool execution result

    Raises:
        HTTPException: If tool execution fails critically
    """
    logger.info(f"Executing tool: {request.tool_name}")

    # Check if tool exists
    if not tool_registry.get_tool(request.tool_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request.tool_name}' not found"
        )

    # Execute tool
    result: ToolResult = executor.execute_tool(
        request.tool_name,
        request.arguments
    )

    return ToolExecutionResponse(
        success=result.success,
        result=result.result,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        retries_used=result.retries_used
    )


@app.post("/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    """
    Process query with ReAct agent.

    Args:
        request: User query and optional parameters

    Returns:
        Agent's answer with reasoning trace

    Note:
        If OpenAI API key not configured, returns skipped response
    """
    logger.info(f"Agent query: {request.query[:50]}...")

    # Check if LLM is configured
    if not config.is_configured("openai"):
        logger.warning("OpenAI not configured, skipping agent query")
        return AgentQueryResponse(
            answer="Agent query skipped - OpenAI API not configured",
            success=False,
            iterations=0,
            trace=[],
            skipped=True,
            reason="no_openai_key"
        )

    # Run agent
    response = agent.run(request.query)

    return AgentQueryResponse(
        answer=response["answer"],
        success=response["success"],
        iterations=response["iterations"],
        trace=response["trace"]
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get tool execution statistics.

    Returns:
        Statistics for all tools (calls, successes, failures, avg time)
    """
    stats = tool_registry.get_stats()
    return StatsResponse(stats=stats)


@app.get("/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint (optional).

    Returns:
        Metrics in Prometheus text format
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"message": "Prometheus client not installed"}


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Application starting up...")
    logger.info(f"Registered tools: {len(tool_registry.list_tools())}")

    # Check configuration
    services = config.validate_config()
    for service, configured in services.items():
        status_icon = "✅" if configured else "⚠️"
        logger.info(f"{status_icon} {service}: {'configured' if configured else 'not configured'}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Application shutting down...")
    executor.shutdown()
    logger.info("Executor shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """Run app locally with uvicorn."""
    import sys

    print("=" * 60)
    print("Module 10.2: Tool Calling & Function Execution API")
    print("=" * 60)
    print(f"\n🚀 Starting FastAPI server...")
    print(f"📊 Registered tools: {len(tool_registry.list_tools())}")
    print(f"\n📝 Endpoints:")
    print("  GET  /health          - Health check")
    print("  GET  /tools           - List available tools")
    print("  POST /tool/execute    - Execute single tool")
    print("  POST /query           - Agent query with ReAct loop")
    print("  GET  /stats           - Tool execution statistics")
    print("  GET  /docs            - Interactive API documentation")
    print(f"\n⚠️  Note: Some features require API keys (see .env.example)")
    print("=" * 60 + "\n")

    # Run server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
