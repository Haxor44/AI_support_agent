# AI Customer Support Agent on Amazon Bedrock AgentCore

A customer support agent for a fictional online store, deployed on Amazon Bedrock AgentCore. It handles multi-step requests such as order tracking, refunds, product questions and loyalty discounts, and remembers customers across sessions.

Built as part of the Udacity AWS AI Engineering Nanodegree. The order and refund Lambda functions were provided by the course.

## What it does

- **Answers product and policy questions** using RAG over a Bedrock Knowledge Base
- **Tracks orders and processes refunds** through Lambda tools exposed via AgentCore Gateway (MCP)
- **Remembers customers** with short-term session memory and long-term memory for facts and preferences
- **Calculates loyalty discounts** in a sandboxed Code Interpreter, with a fallback if it is unavailable
- **Looks up live web pages** with the AgentCore Browser tool

## Architecture

```
Customer request
       │
AgentCore Runtime ── Strands agent (Amazon Nova 2 Lite)
       │
       ├── AgentCore Memory ────── customer facts and preferences
       ├── Bedrock Knowledge Base ─ product catalog (RAG)
       ├── Code Interpreter ─────── loyalty discount calculations
       ├── AgentCore Browser ────── live web lookups
       └── AgentCore Gateway (MCP)
              ├── order-tracker Lambda (via API Gateway)
              └── refund-processor Lambda
```

## Tools

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Retrieves product and policy information |
| `calculate_loyalty_discount` | Calculates discounts from tier, points and order total |
| Order tracker (Gateway) | Looks up orders and customer profiles |
| `initiate_refund`, `check_refund_status`, `get_return_label` (Gateway) | Handles refunds and returns |
| AgentCore Browser | Fetches information from live websites |

## Tech stack

Python, Strands Agents SDK, Amazon Bedrock AgentCore (Runtime, Gateway, Memory, Code Interpreter, Browser), Bedrock Knowledge Bases, Amazon Nova, AWS Lambda, API Gateway, MCP

## Project structure

```
starter/
├── main.py               Agent, tools, memory hook and entry point
├── lambda/               Order and refund functions, plus the tool schema
├── product_catalog.txt   Knowledge Base source document
├── reflection.md         Design decisions and lessons learned
├── screenshots/          Test runs against the deployed agent
└── pyproject.toml        Dependencies
```

## Running it

You need an AWS account with Bedrock model access, the AWS CLI, Python and [uv](https://docs.astral.sh/uv/). Create the Lambda functions, Gateway, Knowledge Base and Memory resource, then set their values at the top of `starter/main.py`:

```python
GATEWAY_ENDPOINT, KNOWLEDGE_BASE_ID, AWS_REGION, LTM_MEMORY_ID
```

Then deploy and test:

```bash
cd starter
uv sync
uv run agentcore configure
uv run agentcore deploy
uv run agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```

## Testing

I tested six scenarios against the deployed agent. The results are in [`starter/screenshots`](starter/screenshots):

1. Order tracking
2. Refund processing
3. Knowledge Base questions
4. Long-term memory across sessions
5. Loyalty discount calculation
6. Live web lookup

## What I learned

The hardest problem was a browser tool that failed because of missing permissions, not faulty logic. Tracing the failure found the cause, and a small test confirmed the fix. For production, I would add structured logging for each tool call, permission-based access controls, retries, and monitoring for latency, errors and tool usage.
