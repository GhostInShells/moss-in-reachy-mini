import asyncio
import datetime
import os
import time
from typing import List

from anthropic.types.beta import BetaThinkingConfigEnabledParam
from ghoshell_moss.core.concepts.command import CommandTaskResult
from litellm.proxy.proxy_server import model_settings
from pydantic import BaseModel, Field
from pydantic_ai import Tool, RunContext, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai_backends import LocalBackend
from pydantic_deep import create_deep_agent, create_default_deps

from framework.apps.volc_websearch import VolcWebsearchChannel

websearch_chan = VolcWebsearchChannel(
    name="volc_websearch",
    description="火山云搜索",
    api_key="ZD9InVAsVKTUcWWnQaTR2VRChxxN2fQL"
)

async def websearch(ctx: RunContext, query: str) -> CommandTaskResult:
    """Get the latitude and longitude of a location.

    Args:
        ctx: The context.
        query: A description of a location.
    """
    # NOTE: the response here will be random, and is not related to the location description.
    return await websearch_chan.websearch(query)

class TaskOutput(BaseModel):
    filenames: List[str] = Field(description="The filename of the output file")
    summary: str = Field(description="The summary of the task")
    tips: List[str] = Field(description="The tips of the task")

async def main():
    deps = create_default_deps(backend=LocalBackend())
    agent = create_deep_agent(
        model="anthropic:doubao-seed-1-6-251015",
        include_todo=True,          # Task planning
        include_filesystem=True,    # File read/write/edit/execute
        include_subagents=True,     # Delegate to subagents
        include_skills=True,        # Domain-specific skills from SKILL.md files
        # include_memory=True,        # Persistent MEMORY.md across sessions
        # include_plan=True,          # Structured planning before execution
        # include_teams=True,         # Multi-agent teams with shared TODOs
        # include_web=True,           # Web search and URL fetching
        # context_manager=True,       # Auto-summarization for unlimited context
        # cost_tracking=True,         # Token/USD budget enforcement
        # include_checkpoints=True,   # Save, rewind, and fork conversations
        tools=[
            Tool(
                function=websearch,
                name=websearch_chan.name(),
                description=websearch_chan.description(),
            )
        ],
        output_type=TaskOutput,
        model_settings=AnthropicModelSettings(
            anthropic_thinking=BetaThinkingConfigEnabledParam(
                budget_tokens=1000000,
                type="enabled"
            )
        ),
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    result = await agent.run(f"当前时间{now}, 你能帮我制作一份当前伊朗最新局势的新闻稿吗", deps=deps)
    print(result.output)

if __name__ == "__main__":
    os.environ["ANTHROPIC_API_KEY"] = "1c0898fa-e5d9-4a64-9e89-a34dbee0e9e3"
    os.environ["ANTHROPIC_BASE_URL"] = "https://ark.cn-beijing.volces.com/api/compatible"
    asyncio.run(main())
