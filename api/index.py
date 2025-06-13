from fastapi import FastAPI
from mangum import Mangum  # to adapt ASGI to serverless

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI on Vercel"}

handler = Mangum(app)
