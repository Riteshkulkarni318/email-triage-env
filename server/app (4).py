import uvicorn
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
