"""``python -m cropcraft`` / ``cropcraft`` console entrypoint."""

import argparse


def main() -> None:
    """Run the development server."""
    import uvicorn

    parser = argparse.ArgumentParser(prog="cropcraft", description="Run the CropCraft server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()

    uvicorn.run("cropcraft.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
