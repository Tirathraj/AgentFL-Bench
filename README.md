Streamlit UI for the PICU Federated Learning Fairness Pipeline.

**Ollama**

```
ollama pull llama3.2:3b
```

**Conda installation**

```
conda create --name justine  python==3.11
conda activate justine
pip install -r requirements.txt
```



Run:

```
streamlit run app.py
```


Six-step workflow:

1. Doctor asks a question
2. Agent analyzes FL results
3. Agent detects unfairness across hospitals / subgroups
4. Agent suggests corrections (FedProx, reweighting)
5. Doctor approves → model retrains automatically
6. Updated results and audit report are displayed