import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from cym_semantic_layer import agent_executor as cym_semantic_layer_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add

add_routes(app, cym_semantic_layer_chain, path="/cym-semantic-layer")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8080 if $PORT is not set

    uvicorn.run(app, host="0.0.0.0", port=port)
