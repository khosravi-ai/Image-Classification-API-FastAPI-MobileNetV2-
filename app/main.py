from fastapi import FastAPI
from app.routers import classify_router

app = FastAPI()
app.include_router(classify_router.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
