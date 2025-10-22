from fastapi import FastAPI

app = FastAPI(title=\"NBA Player Over/Under Lab\")

@app.get(\"/health\")
def health():
    return {\"status\": \"ok\"}
