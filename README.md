# Support Ticket Router — Homework 3

Files: `support_ticket_router.ipynb` is the runnable notebook; `prompt.md` is the technical specification.

1. Open [Google Colab](https://colab.research.google.com/), choose **File → Upload notebook**, and select `support_ticket_router.ipynb`.
2. Use a Python 3 CPU runtime; no GPU is needed.
3. Open **Secrets** (key icon). Add `OPENAI_API_KEY` with your OpenAI API key as the value and enable **Notebook access**. Available OpenAI API credit is required. Never put the key in notebook code, a file, or this chat.
4. Run cells from top to bottom. The notebook installs `langgraph` and `langchain-openai`, draws the graph, runs one demonstration, and prints ten tests with PASS/MISMATCH indicators.

One complete run makes 11 classification calls, one per graph invocation. Automatic retries are disabled. Re-running demo/test cells makes additional calls. The graph drawing uses an external Mermaid rendering service; if it is unavailable, rerun the drawing cell later. Missing-secret errors require checking the secret name and Notebook access.

The notebook ships with no saved outputs or credentials. Live LLM predictions have not been tested locally. No Git repository, commit, push, or publication is required to run it.
