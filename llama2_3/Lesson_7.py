#!/usr/bin/env python
# coding: utf-8

# # Lab 7: Llama Stack

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from dotenv import load_dotenv
import os
_ = load_dotenv() #loads 'TOGETHER_API_KEY'


# In[3]:


#!pip install llama-stack==0.0.36 llama-stack-client==0.0.35


# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>utils.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# In[4]:


get_ipython().system('llama stack build --list-templates')


# In[5]:


get_ipython().system('llama stack list-apis')


# #  Llama Stack Inference

# In[6]:


LLAMA_STACK_API_TOGETHER_URL="https://llama-stack.together.ai"
LLAMA31_8B_INSTRUCT = "Llama3.1-8B-Instruct"

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage

async def run_main():
    client = LlamaStackClient(
        base_url=LLAMA_STACK_API_TOGETHER_URL,
    )

    iterator = client.inference.chat_completion(
        messages=[
            UserMessage(
                content="Who wrote the book Innovator's Dilemma? How about Charlotte's Web?",
                role="user",
            ),

            UserMessage(
                content="which book was published first?",
                role="user",
            ),
        ],
        model=LLAMA31_8B_INSTRUCT,
        stream=True
    )

    async for log in EventLogger().log(iterator):
        log.print()
        #print("?")

await run_main()


# # Llama Stack Agent

# In[7]:


import asyncio
from typing import List, Optional, Dict

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger

from llama_stack_client.types import SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig


# In[8]:


class Agent:
    def __init__(self):
        self.client = LlamaStackClient(
            base_url=LLAMA_STACK_API_TOGETHER_URL,
        )

    def create_agent(self, agent_config: AgentConfig):
        agent = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = agent.agent_id
        session = self.client.agents.sessions.create(
            agent_id=agent.agent_id,
            session_name="example_session",
        )
        self.session_id = session.session_id

    async def execute_turn(self, content: str):
        response = self.client.agents.turns.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=True,
        )

        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk



# In[9]:


async def run_main():
    agent_config = AgentConfig(
        model=LLAMA31_8B_INSTRUCT,
        instructions="You are a helpful assistant",
        enable_session_persistence=False,
    )

    agent = Agent()
    agent.create_agent(agent_config)

    prompts = [
        "Who wrote the book Charlotte's Web?",
        "Three best quotes?",
    ]

    for prompt in prompts:
        print(f"User> {prompt}")
        response = agent.execute_turn(content=prompt)
        async for log in EventLogger().log(response):
            if log is not None:
                log.print()

await run_main()


# # Llama Stack with Llama 3.2 vision model

# In[10]:


from PIL import Image
import matplotlib.pyplot as plt

def display_image(path):
  img = Image.open(path)
  plt.imshow(img)
  plt.axis('off')
  plt.show()

display_image("./content/Llama_Repo.jpeg")


# In[11]:


import base64

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import agent_create_params

LLAMA32_11B_INSTRUCT = "Llama3.2-11B-Vision-Instruct"

def encode_image(image_path):
  with open(image_path, "rb") as img:
    return base64.b64encode(img.read()).decode('utf-8')


# In[12]:


class Agent:
    def __init__(self):
        self.client = LlamaStackClient(
            base_url=LLAMA_STACK_API_TOGETHER_URL,
        )

    def create_agent(self, agent_config: AgentConfig):
        agent = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = agent.agent_id
        session = self.client.agents.sessions.create(
            agent_id=agent.agent_id,
            session_name="example_session",
        )
        self.session_id = session.session_id

    async def execute_turn(self, prompt: str, image_path: str):
        base64_image = encode_image(image_path)

        messages = [{
            "role": "user",
            "content": [
              {
                "image": {
                  "uri": f"data:image/jpeg;base64,{base64_image}"
                }
              },
              prompt,
            ]
        }]

        response = self.client.agents.turns.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages = messages,
            stream=True,
        )

        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk


# In[13]:


async def run_main(image_path, prompt):
    agent_config = AgentConfig(
        model=LLAMA32_11B_INSTRUCT,
        instructions="You are a helpful assistant",
        enable_session_persistence=False,
    )

    agent = Agent()
    agent.create_agent(agent_config)

    print(f"User> {prompt}")
    response = agent.execute_turn(prompt=prompt, image_path=image_path)
    async for log in EventLogger().log(response):
        if log is not None:
            log.print()



# In[14]:


await run_main("./content/Llama_Repo.jpeg",
         "How many different colors are those llamas?\
         What are those colors?")


# In[ ]:




