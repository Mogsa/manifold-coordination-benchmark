"""Run the ChaosBench API with uvicorn."""

from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("chaosbench.api.app:app", host="0.0.0.0", port=8000, reload=False)
