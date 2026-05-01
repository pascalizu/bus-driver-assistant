from langchain_ollama import ChatOllama

# Use the model you downloaded
llm = ChatOllama(model="llama3.1:8b")

print("Bus Driver Assistant Chat (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    
    response = llm.invoke(user_input)
    print("Assistant:", response.content)