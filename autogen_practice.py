from autogen_agentchat.agents import AssistantAgent 
from autogen_agentchat.ui import Console 
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-40", #api_key="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", 
)

async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"THe weather in {city} is 73 degrees and Sunny."

agent = AssistantAgent(name="weather_agent", model_client=model_client, tools=[get_weather], system_message="You are a helpful assistant.", reflect_on_tool_use=True, model_client_stream=True,)

async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))

 