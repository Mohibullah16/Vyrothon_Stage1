"""
Evaluation Harness Contract
This is the exact interface the grader will call.
No network imports are allowed.
"""

class Agent:
    def __init__(self, model_path: str):
        """
        Initialize the model.
        """
        pass
        
    def predict(self, prompt: str) -> str:
        """
        Given a user prompt, return the assistant's response.
        If it's a tool call, return the JSON string: {"tool": "...", "args": {...}}
        """
        raise NotImplementedError("Grader interface not implemented")

def get_model():
    return Agent("path/to/model")
