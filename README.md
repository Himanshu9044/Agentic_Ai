# 🤖 LangGraph Tool Calling Agent

An AI-powered agent built using **LangGraph**, **LangChain**, **Groq LLM**, and **Tavily Search**. The agent can answer general questions, search the web for up-to-date information, and execute custom tools such as multiplication using LangGraph's tool-calling workflow.

---

## 🚀 Features

- 🧠 Conversational AI using Groq LLM
- 🌐 Web search with Tavily Search API
- 🛠️ Custom tool support (e.g., multiplication)
- 🔄 Automatic tool calling using LangGraph
- 📊 State management with LangGraph
- 🖼️ Mermaid graph generation for workflow visualization

---

## 🛠️ Tech Stack

- Python 3.11+
- LangGraph
- LangChain
- LangChain Groq
- LangChain Tavily
- Python Dotenv
- Mermaid Graph

---

## 📂 Project Structure

```text
LangGraph-Agent/
│
├── app.py                # Main application
├── .env                  # API keys (not committed)
├── .gitignore
├── requirements.txt
├── README.md
└── graph.png             # Generated graph visualization
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/LangGraph-Agent.git
cd LangGraph-Agent
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## ▶️ Run the Project

```bash
python app.py
```

---

## 🧠 How It Works

1. User enters a question.
2. LangGraph sends the message to the Groq LLM.
3. If the LLM requires external information, it automatically calls the Tavily Search tool.
4. If a mathematical operation is required, it calls the custom multiplication tool.
5. The tool result is returned to the LLM.
6. The LLM generates the final response.

---

## 🔄 Workflow

```text
START
   │
   ▼
LLM Node
   │
   ├──────────────┐
   │              │
   ▼              ▼
Tool Needed?     END
   │
   ▼
Tool Node
   │
   ▼
LLM Node
   │
   ▼
END
```

---

## 🧩 Example Queries

### General Chat

```
Hello, how are you?
```

### Web Search

```
What are today's AI news headlines?
```

### Custom Tool

```
Multiply 125 by 48.
```

---

## 📸 Graph Visualization

The application automatically generates a Mermaid graph and saves it as:

```
graph.png
```

---

## 📦 Dependencies

```
langgraph
langchain
langchain-groq
langchain-tavily
python-dotenv
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- Memory support
- Multiple tool integration
- Calculator tool
- Weather API integration
- PDF generation
- Streamlit web interface
- Multi-agent architecture
- Conversation history
- Human-in-the-loop support

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Himanshu Kumar**

- AI/ML Engineer
- B.Tech CSE (AI/ML)
- Passionate about Generative AI, Agentic AI, LangGraph, and Large Language Models.