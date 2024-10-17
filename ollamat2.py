import ollama

stream = ollama.chat(
    model='llama3.1',
    messages=[{'role': 'user', 'content': '为什么天空是蓝色的？'}],
    stream=True,
)

for chunk in stream:
  
  print(chunk['message']['content'], end='', flush=True)
  print(chunk)

