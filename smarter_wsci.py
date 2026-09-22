from pathlib import Path
from ollama import chat
import json



question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

## WRITE ##
service_status = {
    "wifi": "operational"
}

state = {
    "problem": question,
    "wi_fi status": "operational",
    "wi-fi_check": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print(state)

## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
KEYWORD_MAP = {
    "printer":   ["knowledge/printing.txt"],
    "print":     ["knowledge/printing.txt"],
    "projector": ["knowledge/classroom_projectors.txt"],
    "project":   ["knowledge/classroom_projectors.txt"],
    "display":   ["knowledge/classroom_projectors.txt"],

    "password":  ["knowledge/password_changes.txt",
                  "knowledge/wifi_setup.txt",
                  "knowledge/email_setup.txt"],
    "wi-fi":     ["knowledge/wifi_setup.txt",
                  "knowledge/service_status.txt"],
    "wifi":      ["knowledge/wifi_setup.txt",
                  "knowledge/service_status.txt"],
    "eduroam":   ["knowledge/wifi_setup.txt",
                  "knowledge/service_status.txt"],
    "email":     ["knowledge/email_setup.txt"],
    "vpn":       ["knowledge/vpn.txt"],
    "status":    ["knowledge/service_status.txt"],
}

def select_context(question):
    q = question.lower()
    selected = set()
    for keyword, files in KEYWORD_MAP.items():
        if keyword in q:
            selected.update(files)
    return sorted(selected)

selected_files = select_context(question)
print("Selected files:", selected_files)

## READ SELECTED FILES and add their contents to the context variable.
context = ""
for file_path in selected_files:
    p = Path(file_path)
    if p.exists():
        context += p.read_text() + "\n\n"

print("Raw context characters:", len(context))

## 
## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context" 
def compress_context(context, question):
    response = chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a context compressor. "
                    "Extract ONLY the information relevant to the user's question. "
                    "Be concise. Output plain text, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Full context:\n{context}\n\n"
                    "Compressed relevant context:"
                ),
            },
        ],
    )
    return response.message.content.strip()


compressed_context = compress_context(context, question)

## Print the length of the compressed context
print(len(compressed_context))

## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output 
response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT helpdesk assistant. "
                "Answer the student's question using ONLY the provided context. "
                "Respond with JSON containing keys: "
                '"diagnosis", "steps" (list), "confidence" (0-1).'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{compressed_context}\n\n"
                f"Student question:\n{question}"
            ),
        },
    ],
)

print(response.message.content)

## WRITE the above output in an artifact called "state"
try:
    answer = json.loads(response.message.content)
except json.JSONDecodeError:
    answer = {"raw": response.message.content}

state["diagnosis"] = answer.get("diagnosis")
state["steps"] = answer.get("steps", [])
state["confidence"] = answer.get("confidence")

with open("state.json", "w") as file:
    json.dump(state, file, indent=2)

print("Updated state.json:", state)

## Update the rest of the code so that it uses the "state" artifact as part of the context.
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.
diagnostic_context = {
    "problem": question,
    "device": "Windows laptop",
    "wifi_status": "operational",
    "phone_works": True,
}

report_context = {
    "total_wifi_cases": 37,
    "resolved_cases": 29,
    "unresolved_cases": 8,
}

classify = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "Classify the user's issue into one of: 'diagnostic', 'report'. "
                "Reply with only the single word."
            ),
        },
        {"role": "user", "content": question},
    ],
)

task = classify.message.content.strip().lower()
print("Task classified by Qwen:", task)

if task == "report":
    active_state = report_context
else:
    active_state = diagnostic_context

print("Active state:", active_state)

final = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT helpdesk assistant. "
                "Use ONLY the provided state to answer the student's question. "
                "Do not invent information that is not in the state."
            ),
        },
        {
            "role": "user",
            "content": (
                f"State:\n{json.dumps(active_state, indent=2)}\n\n"
                f"Student question:\n{question}"
            ),
        },
    ],
)

print("Final answer using isolated state:")
print(final.message.content)