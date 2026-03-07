# 框架支持
[MOSShell](https://github.com/GhostInShells/MOSShell)

# Start

## Setup venv
```
uv venv
source .venv/bin/activate
uv sync
uv pip install --index-url https://gitlab.freedesktop.org/api/v4/projects/1340/packages/pypi/simple gstreamer==1.28.0
```

## Setup .env
```commandline
cp .env.example .env
```

## Run Reachy Mini
[Reachy Mini README](./src/moss_in_reachy_mini/README.md)