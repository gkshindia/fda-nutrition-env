import uvicorn
from env.server.app import app  # noqa: F401


def main():
    uvicorn.run("env.server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
