"""Agents nodes — one node class per file. Re-exported here so
PluginContext.discover_nodes() finds them via inspect.getmembers.
"""
from nodes.agents.ollama_client import OllamaClientNode
from nodes.agents.openai_compat_client import OpenAICompatClientNode
from nodes.agents.prompt_template import PromptTemplateNode
from nodes.agents.chat_message import ChatMessageNode
from nodes.agents.conversation import ConversationNode
from nodes.agents.agent import AgentNode
from nodes.agents.tool import ToolNode
from nodes.agents.python_function_tool import PythonFunctionToolNode
