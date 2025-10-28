"""Simple Flask app demonstrating a RAG-style chatbot scaffold."""
from flask import Flask, request, jsonify, render_template_string
import os
from scripts.rag_inference import RAGSystem

app = Flask(__name__)
rag = RAGSystem(index_dir='models/index')

INDEX_HTML = '''
<!doctype html>
<title>RAG Chatbot</title>
<h1>RAG Chatbot (scaffold)</h1>
<form method=post action="/chat">
  <input name="query" style="width:80%" placeholder="Ask something...">
  <input type=submit value=Send>
</form>
<div id="response">{{response}}</div>
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(INDEX_HTML, response='')

@app.route('/chat', methods=['POST'])
def chat():
    user_q = request.form.get('query')
    answer = rag.answer(user_q)
    return render_template_string(INDEX_HTML, response=answer)

if __name__ == '__main__':
    app.run(debug=True)
