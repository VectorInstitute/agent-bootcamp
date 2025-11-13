# 2.4 Agents with access to MCP Servers in OpenAI Agents SDK

This folder introduces use of Model Context Protocol (MCP) Servers to allow agents to access data and tools. The `mcp-server-git` MCP server is provided to the agent with limited tool use so it can use `git` commands in the repo.


# Running

```bash
uv run --env-file .env python3 src/2_frameworks/4_mcp/app.py
```


