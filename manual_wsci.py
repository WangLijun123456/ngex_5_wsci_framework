from pathlib import Path
from ollama import chat

question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

selected_files = [
    ##Use only the files that are relevant to the question.
    "knowledge/password_changes.txt",
    "knowledge/wifi_setup.txt",
    "knowledge/service_status.txt",
]

context = ""

## Write a for loop to go through all the files in selected_files and read their contents into the context variable.
## Call Qwen with the student's question and the context you created above.
for file_path in selected_files:
    context += Path(file_path).read_text()
    context += "\n\n"

response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT helpdesk assistant. "
                "Use the provided knowledge base context to answer the student's question."
            ),
        },
        {
            "role": "user",
            "content": f"Knowledge base:\n{context}\n\nStudent question:\n{question}",
        },
    ],
)

print(
    "Context characters:",
    len(context)
)
print(response.message.content)





