"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the step-by-step solution files and INSTRUCTIONS.md for guidance.
Do NOT copy the solution directly — work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json, re
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("CSAI_Agent")
logger.setLevel(logging.DEBUG)

# ── TODO 1 — App Initialisation ───────────────────────────────────────────────
# Create a BedrockAgentCoreApp instance.
# This registers the ASGI server for AgentCore deployment.
# There must be exactly one instance per deployment.
#
# Hint: app = BedrockAgentCoreApp()

# TODO: Create the BedrockAgentCoreApp instance
app = BedrockAgentCoreApp()  # Replace this line


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ── TODO 2 — Configuration ────────────────────────────────────────────────────
# Replace the placeholder strings with your actual AWS resource values.
# You collected these in Part 1 of the INSTRUCTIONS.
#
# GATEWAY_URL format: https://<alias>.gateway.bedrock-agentcore.<region>.amazonaws.com/mcp
# KB_ID       format: 10-character alphanumeric string from the KB console
# REGION:     your AWS region, e.g. "us-east-1"
# MEMORY_ID   format: shown in the AgentCore Memory console


GATEWAY_ENDPOINT = "https://customersupportgateway-tvwxoqd5is.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"   # TODO: Replace with your Gateway URL
KNOWLEDGE_BASE_ID = "7OGE2PVPJV"          # TODO: Replace with your Knowledge Base ID
AWS_REGION      = "us-east-1"        # TODO: Replace with your AWS region
LTM_MEMORY_ID   = "CustomerSupportMemory-veKt8fBPdR" 


# ── TODO 3 — Model and Clients ────────────────────────────────────────────────
# Create:
#   1. A BedrockModel using model_id "global.amazon.nova-2-lite-v1:0"
#   2. A MemoryClient with region_name=REGION
#   3. A boto3 client for the "bedrock-agent-runtime" service in REGION
#
# Hint: model = BedrockModel(model_id=model_id)

MODEL_IDENTIFIER = "global.amazon.nova-2-lite-v1:0"
# TODO: Create the BedrockModel instance
llm_model = BedrockModel(model_id=MODEL_IDENTIFIER)

# TODO: Create the MemoryClient instance
mem_client = MemoryClient(region_name=AWS_REGION)

# TODO: Create the boto3 bedrock-agent-runtime client
bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)


# ── TODO 4 — Namespace Helper ─────────────────────────────────────────────────
# Implement get_namespaces() to return a dict mapping strategy type to
# namespace template string.
#
# Steps:
#   1. Call mem_client.get_memory_strategies(memory_id) to get strategy list
#   2. Return a dict: { strategy["type"]: strategy["namespaces"][0] for each strategy }
#
# Example output:
#   { "SEMANTIC": "cs_agent/{actorId}/facts",
#     "USER_PREFERENCE": "cs_agent/{actorId}/preferences" }

def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    # Keyed by name, not type — custom-override strategies all report the
    # same generic "CUSTOM" type, so keying on type drops every strategy
    # but the last.
    return {strategy["name"]: strategy["namespaces"][0] for strategy in strategies}


# ── TODO 5 — Memory Hook ──────────────────────────────────────────────────────
# Implement MemoryHook, a HookProvider subclass that adds long-term memory.
#
# The class needs:
#   __init__(self, actor_id, session_id, memory_client, memory_id)
#     — store all four as instance attributes
#     — call get_namespaces() and store the result as self.namespaces
#
#   retrieve_customer_context(self, event: MessageAddedEvent)
#     — only runs for plain-text user messages (not tool results)
#     — for each strategy namespace, call memory_client.retrieve_memories(
#          memory_id, namespace (formatted with actorId), query, top_k=5)
#     — collect non-empty memory texts tagged with their strategy type
#     — if any memories found, prepend them to the user message as:
#          "Customer Context:\n<memories>\n\n<original_message>"
#
#   save_support_interaction(self, event: AfterInvocationEvent)
#     — walk the message list backwards to find the last plain-text user
#       query and the last assistant response
#     — call memory_client.create_event(memory_id, actor_id, session_id,
#          messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")])
#
#   register_hooks(self, registry: HookRegistry)
#     — register retrieve_customer_context on MessageAddedEvent
#     — register save_support_interaction on AfterInvocationEvent

class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(self.memory_client, self.memory_id)
        self._last_raw_query = None

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        messages = event.agent.messages
        if not messages:
            return

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return

        content = last_message.get("content", [])
        if not content or "text" not in content[0]:
            return

        user_query = content[0]["text"]
        # Cache the raw query for save_support_interaction() to persist,
        # since a <user_context> block may get inserted ahead of it below.
        self._last_raw_query = user_query

        memory_context = []
        customer_name = None
        for strategy_name, namespace in self.namespaces.items():
            namespace = namespace.format(actorId=self.actor_id)
            memories = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=namespace,
                query=user_query,
                top_k=5,
            )
            for memory in memories:
                text = memory.get("content", {}).get("text", "")
                if not text:
                    continue
                memory_context.append(f"[{strategy_name}] {text}")
                # Pull a name out of facts like "Jane's name is Jane" so it
                # can be stated plainly rather than left for the model to
                # infer from odd third-person phrasing.
                if customer_name is None:
                    name_match = re.search(r"\bname is ([A-Z][a-zA-Z'-]*)", text)
                    if name_match:
                        customer_name = name_match.group(1)

        if memory_context:
            if customer_name:
                memory_context.insert(
                    0, f"Customer's first name: {customer_name}"
                )
            context_text = "\n".join(memory_context)
            # A separate content block ahead of the customer's own message,
            # not merged into its text — keeps the model's actual request
            # distinguishable from retrieved reference material.
            content.insert(0, {"text": f"<user_context>{context_text}</user_context>"})

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages

        # Prefer the clean query cached in retrieve_customer_context() before
        # it added the <user_context> block, so we save what the customer
        # actually said, not the retrieved context alongside it.
        customer_query = self._last_raw_query
        agent_response = None

        for message in reversed(messages):
            content = message.get("content", [])
            if not content or "text" not in content[0]:
                continue
            if agent_response is None and message.get("role") == "assistant":
                agent_response = content[0]["text"]
            elif customer_query is None and message.get("role") == "user":
                customer_query = content[0]["text"]

            if customer_query is not None and agent_response is not None:
                break

        if customer_query is None or agent_response is None:
            return

        self.memory_client.create_event(
            memory_id=self.memory_id,
            actor_id=self.actor_id,
            session_id=self.session_id,
            messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
        )

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── TODO 6 — Knowledge Base Tool ─────────────────────────────────────────────
# Implement search_knowledge_base(query) using the @tool decorator.
#
# Steps:
#   1. Guard: if KB_ID is empty return "Knowledge base not configured."
#   2. Call _bedrock_runtime.retrieve(
#          knowledgeBaseId=KB_ID,
#          retrievalQuery={"text": query}
#      )
#   3. Extract resp["retrievalResults"]; return a message if empty
#   4. Join the text chunks with "\n---\n" and return the result
#
# The docstring is the tool description — the model uses it to decide when
# to call this tool, so keep it clear and accurate.

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KNOWLEDGE_BASE_ID:
        return "Knowledge base not configured."

    try:
        resp = bedrock_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
        )

        results = resp.get("retrievalResults", [])

    except Exception as e:
        logger.exception("Knowledge base retrieval failed for query: %s", query)
        return f"Knowledge base access error: {e}"
    if not results:
        return f"No knowledge base results found for: {query}"

    chunks = [r["content"]["text"] for r in results]
    return "\n---\n".join(chunks)


# ── TODO 7 — Loyalty Discount Tool (Code Interpreter) ────────────────────────
# Implement calculate_loyalty_discount() using the @tool decorator.
#
# The tool must:
#   1. Build a self-contained Python code string that:
#        • Defines earn_rates: {"standard": 1, "device": 2, "fresh": 5}
#        • Defines tier_rates: {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
#        • Calculates points_redeemed (floor to nearest 500, cap at 50% of order)
#        • Calculates tier_discount (applied to subtotal after points)
#        • Calculates final_total, total_savings, points_earned, remaining_points
#        • Prints a JSON result dict
#   2. Execute the code with code_session(REGION).invoke("executeCode", {...})
#      using language="python" and clearContext=True
#   3. Return the first result event as a JSON string
#   4. Include a fallback that computes only the tier discount if the
#      Code Interpreter is unavailable

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points:   Customer's current points balance
        tier:             Customer tier — Silver, Gold, or Platinum
        order_total:      Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

# Redeem points in $0.01-per-point increments, floored to the nearest 500
# points, capped at 50% of the order total. Work in whole points (not
# dollar amounts) to avoid floating-point division artifacts.
max_redeemable_points = int(round(order_total * 50))
points_redeemed = min(loyalty_points, max_redeemable_points)
points_redeemed = (points_redeemed // 500) * 500
points_discount = round(points_redeemed * 0.01, 2)

subtotal_after_points = order_total - points_discount
tier_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * tier_rate

final_total = subtotal_after_points - tier_discount
total_savings = order_total - final_total

earn_rate = earn_rates.get(product_category, earn_rates["standard"])
points_earned = int(final_total * earn_rate)
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier": tier,
    "tier_discount_pct": int(tier_rate * 100),
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}

print(json.dumps(result))
"""

    try:
        with code_session(AWS_REGION) as code_client:
            response = code_client.invoke("executeCode", {
                "code": code,
                "language": "python",
                "clearContext": True,
            })

        for event in response["stream"]:
            return json.dumps(event["result"])

    except Exception as e:
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = order_total * tier_rate
        final_total = order_total - tier_discount
        return json.dumps({
            "error": f"Code Interpreter unavailable ({e}); tier-discount-only fallback used",
            "order_total": order_total,
            "tier": tier,
            "tier_discount_pct": int(tier_rate * 100),
            "tier_discount": round(tier_discount, 2),
            "final_total": round(final_total, 2),
        })


# ── TODO 8 — Agent Entrypoint ─────────────────────────────────────────────────
# Implement the invoke() function decorated with @app.entrypoint.
#
# Steps:
#   1. Extract user_input, actor_id, and session_id from the payload
#      (generate a UUID if session_id is missing)
#   2. Instantiate MemoryHook for this actor/session
#   3. Instantiate AgentCoreBrowser(region=REGION)
#   4. Build the tools list: [search_knowledge_base, calculate_loyalty_discount,
#                              agent_core_browser.browser]
#   5. Connect to the Gateway via MCPClient, load gateway_tools, extend tools list
#   6. Create and invoke the Agent with all tools, hooks, and system_prompt
#   7. Return the text from the first content block of the response
#   8. Handle exceptions gracefully

@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.

    Expected payload keys:
      prompt      (str, required) — the customer's message
      customer_id (str, optional) — unique customer identifier
      session_id  (str, optional) — session identifier; generated if absent
    """
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=mem_client,
            memory_id=LTM_MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=AWS_REGION)

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser,
        ]

        gateway_client = MCPClient(
            lambda: streamable_http_client(GATEWAY_ENDPOINT)
        )
        with gateway_client:
            try:
                gateway_tools = gateway_client.list_tools_sync()
                tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

            except TimeoutError:
                logger.exception("Gateway tool loading timed out")

            except ConnectionError:
                logger.exception("Gateway connection failed")

            except Exception as exc:
                logger.exception(
                    "Gateway tool loading failed: %s", exc
                )

            agent = Agent(
                model=llm_model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a concise, customer-focused support assistant for an "
                    "online retailer. When handling requests, prefer exact answers "
                    "and use tools to fetch orders, run refunds, consult the "
                    "support knowledge base, compute loyalty discounts, or "
                    "visit external webpages when requested. Maintain a polite, "
                    "professional tone.\n\n"
                    "When a <user_context> block appears, it contains factual "
                    "notes and preferences about this customer (for example: "
                    "name, saved addresses, and prior issues). Use those facts "
                    "directly to improve your reply — include the customer’s "
                    "name when appropriate and prefer concrete, actionable steps."
                ),
            )

            response = agent(user_input)

        return {"result": response.message["content"][0]["text"]}

    except Exception as e:
        logger.exception("Error handling invocation")
        return {"error": str(e)}


# ── CLI entry point (do not modify) ──────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    # main()