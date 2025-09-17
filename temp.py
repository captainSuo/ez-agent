from ez_agent import Agent

agent = Agent(
    base_url="https://api.openai.com/v1/completions",
    api_key="sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    model="text-davinci-003",
    temperature=0.9,
    max_tokens=1000,
    frequency_penalty=0,
    thinking=True,
    message_expire_time=60,
)
