"""Point d'entrée : lance l'API FastAPI avec uvicorn."""
from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
