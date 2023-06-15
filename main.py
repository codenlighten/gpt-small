from pygpt4all.models.gpt4all import GPT4All
import os
import requests

model_dir = './models'
os.makedirs(model_dir, exist_ok=True)
gpt4all_model_url = 'https://huggingface.co/mrgaang/aira/resolve/main/gpt4all-converted.bin'
gpt4all_model_path = os.path.join(model_dir, 'gpt4all-converted.bin')

if not os.path.exists(gpt4all_model_path):
    print("Downloading the GPT4All model...")
    response = requests.get(gpt4all_model_url)
    with open(gpt4all_model_path, 'wb') as f:
        f.write(response.content)
    print("GPT4All model downloaded.")
else:
    print("GPT4All model already exists. No need to download.")

chat_history = ""  # initialize the chat history

model = GPT4All(gpt4all_model_path)

def new_text_callback(text):
    global chat_history
    print(text, end="")
    chat_history += text

user_input = input("\nYou: ")  # get the user input
chat_history += user_input  # add the user input to the chat history
helper_function = ''
while True:
    response3 = model.generate(f"{chat_history}", n_predict=225, new_text_callback=new_text_callback)
    chat_history = response3
    helper_function += chat_history
    # save helper_function to helper_function.txt, if it does not exist, create it
    if len(chat_history) > 475:  # assuming 1 char = 1 token
        chat_history = chat_history[-125:]


    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Goodbye!")
        break