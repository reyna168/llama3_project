#!/usr/bin/env python
# coding: utf-8

# # Llama 3.2 Multimodal Prompting

# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from utils import load_env
load_env()


# ## Text input only question

# In[ ]:


from utils import llama32
from utils import llama31


# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>utils.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# In[ ]:


messages = [
  {"role": "user",
    "content": "Who wrote the book Charlotte's Web?"}
]


# In[ ]:


response_32 = llama32(messages, 90)
print(response_32)


# In[ ]:


response_31 = llama31(messages,70)
print(response_31)


# ## Reprompting with new question

# In[ ]:


messages = [
  {"role": "user",
    "content": "Who wrote the book Charlotte's Web?"},
      {"role": "assistant",
    "content": response_32},
      {"role": "user",
    "content": "3 of the best quotes"}
]


# In[ ]:


response_32 = llama32(messages,90)
print(response_32)


# In[ ]:


response_31 = llama31(messages,70)
print(response_31)


# ## Question about an image

# In[ ]:


from utils import disp_image


# In[ ]:


disp_image("images/Llama_Repo.jpeg") # Example usage for local image


# ### Image from a URL

# In[ ]:


image_url = ("https://raw.githubusercontent.com/meta-llama/"
            "llama-models/refs/heads/main/Llama_Repo.jpeg")
messages = [
  {"role": "user",
    "content": [
      {"type": "text",
        "text": "describe the image in one sentence"
      },
      {"type": "image_url",
        "image_url": {"url": image_url}
      }
    ]
  },
]


# In[ ]:


disp_image(image_url)
result = llama32(messages,90)
print(result)


# ### Using a local image

# In[ ]:


import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
base64_image = encode_image("images/Llama_Repo.jpeg")


# In[ ]:


messages = [
  {"role": "user",
    "content": [
      {"type": "text",
        "text": "describe the image in one sentence"
      },
      {"type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
      }
    ]
  },
]


# In[ ]:


disp_image(image_url)
result = llama32(messages,90)
print(result)


# ## Follow up question about an image

# In[ ]:


messages = [
  {"role": "user",
    "content": [
      {"type": "text",
        "text": "describe the image in one sentence"
      },
      {"type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
      }
    ]
  },
  {"role": "assistant", "content": result},
  {"role": "user", "content": "how many of them are purple?"}
]


# In[ ]:


result = llama32(messages)
print(result)


# ### Define llama32pi() helper

# In[ ]:


def llama32pi(prompt, image_url, model_size=90):
  messages = [
    {
      "role": "user",
      "content": [
        {"type": "text",
          "text": prompt},
        {"type": "image_url",
          "image_url": {
            "url": image_url}
        }
      ]
    },
  ]
  result = llama32(messages, model_size)
  return result


# In[ ]:


print(llama32pi("describe the image in one sentence",
                "https://raw.githubusercontent.com/meta-llama/"
                "llama-models/refs/heads/main/Llama_Repo.jpeg"))


# In[ ]:


print(llama32pi("describe the image in one sentence",
                f"data:image/jpeg;base64,{base64_image}"))


# ## Plant recognition

# In[ ]:


disp_image("images/tree.jpg")


# In[ ]:


question = ("What kind of plant is this in my garden?"
            "Describe it in a short paragraph.")


# In[ ]:


base64_image = encode_image("images/tree.jpg")
result = llama32pi(question, f"data:image/jpg;base64,{base64_image}")
print(result)


# ## Dog breed recognition

# In[ ]:


disp_image("images/ww1.jpg")


# In[ ]:


question = (("What dog breed is this? Describe in one paragraph,"
             "and 3-5 short bullet points"))
base64_image = encode_image("images/ww1.jpg")
result = llama32pi(question, f"data:image/jpg;base64,{base64_image}")
print(result)


# In[ ]:


disp_image("images/ww2.png")


# In[ ]:


base64_image = encode_image("images/ww2.png")
result = llama32pi(question, f"data:image/png;base64,{base64_image}")
print(result)


# ## Tire pressure warning

# In[ ]:


disp_image("images/tire_pressure.png")


# In[ ]:


question = (("What's the problem this is about?"
             " What should be good numbers?"))


# In[ ]:


base64_image = encode_image("images/tire_pressure.png")
result = llama32pi(question, f"data:image/png;base64,{base64_image}")
print(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




