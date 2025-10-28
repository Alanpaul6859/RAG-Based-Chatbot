# Prompt Engineering Guide

Example prompt template (few-shot):

System: You are a helpful assistant. Use only the provided context to answer.

Context:
{context}

User: {user_query}
Assistant:

Tips:
- Keep instructions explicit.
- Ask clarifying questions when user intent is ambiguous.
- If answer is not in context, say you don't know instead of hallucinating.
