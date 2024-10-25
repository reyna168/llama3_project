import os
from groq import Groq

# 调用groq的api文本生成，groq文档：https://console.groq.com/docs/quickstart
def get_ai_response(user_message):
    client = Groq(
        api_key="",
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
        model="gemma-7b-it",     # 模型有llama2-70b-4096、mixtral-8x7b-32768、gemma-7b-it
        # temperature=0.5,
        # max_tokens=1024,

        # Controls diversity via nucleus sampling: 0.5 means half of all
        # likelihood-weighted options are considered.
        # top_p=1,

        # A stop sequence is a predefined or user-specified text string that
        # signals an AI to stop generating content, ensuring its responses
        # remain focused and concise. Examples include punctuation marks and
        # markers like "[end]".
        # stop=None,

        # If set, partial message deltas will be sent.
        # stream=False, 
    )
    return chat_completion.choices[0].message.content

if __name__ == '__main__':
    print(get_ai_response("你能干吗？"))