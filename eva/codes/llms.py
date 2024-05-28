
from openai import OpenAI


def callChatGPT_hot(prompt, model_name, client):
  completion = client.chat.completions.create(
    model=model_name,
    # model="gpt-4",
    messages=[
      {"role": "system", "content": "Requirements: you must choose 10 items from the candidate list instead of the user history list for recommendation and sort them in order of priority, from highest to lowest. Output format: a python list of item's title. Do not explain the reason or include any other words. Do not truncate the list."},
      {"role": "user", "content": prompt}
    ],
  )
#   print(completion.choices[0].message.content)
  return completion

def callChatGPT(prompt, model_name, client):
  completion = client.chat.completions.create(
    model=model_name,
    # model="gpt-4",
    messages=[
      {"role": "system", "content": "Requirements: you must choose 10 items from the candidate list instead of the user history list for recommendation and sort them in order of priority, from highest to lowest. Output format: a python list of item's title. Do not explain the reason or include any other words. Do not truncate the list."},
      {"role": "user", "content": prompt}
    ],
    temperature=0,
  )
#   print(completion.choices[0].message.content)
  return completion



def callChatGLM_hot(prompt, model_name, client):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Requirements: you must choose 10 items from the candidate list instead of the user history list for recommendation and sort them in order of priority, from highest to lowest. Output format: a python list of item's title. Do not explain the reason or include any other words. Do not truncate the list."},
            # {"role": "system", "content": "You are a smart recommender system. Now your task is: given a list of user history, please generate a user profile of this specific user as accuracy and detailed as you can, since it will beneficial to the recommendation accuracy. And give me a summary of the user profile at the last paragraph."},
            {"role": "user", "content": prompt}
        ],
    )
    return response

def callChatGLM(prompt, model_name, client):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Requirements: you must choose 10 items from the candidate list instead of the user history list for recommendation and sort them in order of priority, from highest to lowest. Output format: a python list of item's title. Do not explain the reason or include any other words. Do not truncate the list."},
            # {"role": "system", "content": "You are a smart recommender system. Now your task is: given a list of user history, please generate a user profile of this specific user as accuracy and detailed as you can, since it will beneficial to the recommendation accuracy. And give me a summary of the user profile at the last paragraph."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.01,
    )
    return response


def callLLM_hot(prompt, model_name):
  if model_name == 'gpt-3.5-turbo' or model_name == 'gpt-4':
    client = OpenAI()
    return callChatGPT_hot(prompt, model_name, client)
  elif model_name == 'glm-4':
    client = ZhipuAI(api_key="xxxxxxxxxxx")
    return callChatGPT_hot(prompt, model_name, client)
  else:
    raise NotImplementedError
  
def callLLM(prompt, model_name):
  if model_name == 'gpt-3.5-turbo' or model_name == 'gpt-4':
    client = OpenAI()
    return callChatGPT(prompt, model_name, client)
  elif model_name == 'glm-4':
    client = ZhipuAI(api_key="xxxxxxxxxxx")
    return callChatGPT(prompt, model_name, client)
  else:
    raise NotImplementedError