# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **MOSS in Reachy Mini** - an integration of the MOSS conversational AI framework with the Reachy Mini robot platform. The system enables natural language interaction with a physical robot, supporting multiple operational modes including:

- **Desktop companion mode**: Default wake/sleep states with face tracking and basic interactions
- **Live streaming mode**: Douyin (TikTok) live streaming integration with barrage (chat) interaction
- **Teaching mode**: Programmable movement sequences via conversational commands

The architecture is built on `ghoshell_moss` for agent communication, uses dependency injection via a custom IoC container, and implements a state machine for robot behavior control.

## Setup and Environment

### Prerequisites
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Reachy Mini robot hardware (for physical operation)
- VolcEngine API credentials for LLM and TTS (see `.env.example`)

### Initial Setup
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# Install optional dependencies for live streaming
uv sync --all-extras

# Install GStreamer for audio processing
uv pip install --index-url https://gitlab.freedesktop.org/api/v4/projects/1340/packages/pypi/simple gstreamer==1.28.0

# Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences
```

### Key Environment Variables
- `MOSS_LLM_BASE_URL`, `MOSS_LLM_MODEL`, `MOSS_LLM_API_KEY`: LLM configuration
- `USE_VOICE_SPEECH`: Enable/disable audio responses ("yes"/"no")
- `VOLCENGINE_STREAM_TTS_APP`, `VOLCENGINE_STREAM_TTS_ACCESS_TOKEN`: TTS service credentials
- `REACHY_MINI_MODE`: Operational mode ("live" for streaming, empty for desktop mode)
- `NEWSAPI_API_KEY`: News API access (optional)

## Common Development Commands

### Running the Application
```bash
# Default desktop mode
python src/moss_in_reachy_mini/main.py

# Live streaming mode (requires REACHY_MINI_MODE="live" in .env)
REACHY_MINI_MODE="live" python src/moss_in_reachy_mini/main.py
```

### Memory Management UI
```bash
streamlit run src/moss_in_reachy_mini/scripts/memory_ui.py
```

### Face Recognition Training
```bash
# Add face images to .workspace/runtime/vision/faces/<person_name>/
python src/moss_in_reachy_mini/scripts/train_face.py
```

### Video Recording
```bash
# Demo video recording integration
python src/moss_in_reachy_mini/scripts/ctml_video_recorder_demo.py

# Audio device information
python src/moss_in_reachy_mini/scripts/ouput_audio_info.py
```

### Testing
```bash
# Install development dependencies
uv sync --group dev

# Run all tests
pytest

# Run specific test file
pytest tests/framework/agent/test_storage_memory.py

# Run with verbose output
pytest -v
```

### Linting and Formatting
```bash
# Check code with ruff
ruff check .

# Auto-fix fixable issues
ruff check --fix .

# Format code with ruff
ruff format .
```

### Dependency Management
- Main dependencies: `pyproject.toml`
- Lock file: `uv.lock`
- Local wheel: `dependencies/ghoshell_moss-0.1.0a0-py3-none-any.whl`
- Install with extras: `uv sync --all-extras`

## Architecture

### Core Components

1. **MossInReachyMini** (`src/moss_in_reachy_mini/moss.py`):
   - Central integration point between MOSS framework and Reachy Mini
   - Manages state machine and component lifecycle
   - Provides PyChannel for agent communication

2. **State Machine** (`src/moss_in_reachy_mini/state/`):
   - `AsleepState`: Low-power idle state
   - `WakenState`: Active interaction state with face tracking
   - `BoringState`: Attention-seeking behavior
   - `LiveState`: Douyin live streaming integration (optional)
   - `TeachingState`: Programmable movement recording
   - States implement `MiniStateHook` interface for lifecycle hooks

3. **Robot Components** (`src/moss_in_reachy_mini/components/`):
   - `Body`: Dance and emotion movements
   - `Head`: Head movement and face tracking
   - `Antennas`: Antenna (ear) motor control
   - `Vision`: Camera feed and face recognition

4. **Framework Layer** (`src/framework/`):
   - `AgentHub`: Multi-agent coordination with event bus
   - `MOSSShell`: Core shell implementation from ghoshell_moss
   - `Container`: Dependency injection infrastructure
   - Apps: `live`, `memory`, `session`, `news`, `todolist`

5. **Peripheral Services**:
   - `Listener`: Audio input processing with VAD and ASR
   - `Camera`: Camera feed processing and annotation
   - `Audio`: Microphone hub and stream player
   - `Video`: Recording worker for session capture

### Key Design Patterns

- **Dependency Injection**: All components registered in IoC container via Providers
- **Event-Driven Architecture**: `EventBus` for inter-agent communication
- **Channel-Based Communication**: PyChannels for tool exposure to LLM agents
- **State Pattern**: Robot behavior controlled by state implementations
- **Provider Pattern**: Factory classes for container registration

### Important Directories

- `src/moss_in_reachy_mini/`: Robot-specific integration code
- `src/framework/`: Reusable agent framework components
- `src/framework/apps/`: Application modules (live, memory, etc.)
- `src/framework/abcd/`: Abstract base classes and interfaces
- `.workspace/runtime/`: Runtime storage (memory, sessions, face data)
- `.workspace/configs/`: Configuration files (live streaming settings)

### Container and Dependency Management

The system uses a hierarchical container system:
- Root container: Common dependencies and shared services
- Agent containers: Isolated per-agent contexts with parent reference
- Providers: Register factory methods and lifecycle hooks

Common dependencies include `ReachyMini`, `StorageMemory`, `DouyinLive`, and various component providers.

## Development Notes

### Adding New States
1. Create state class implementing `MiniStateHook` interface
2. Add `NAME` class constant (lowercase)
3. Implement `on_self_enter()` and `on_self_exit()` methods
4. Register provider in `common_dependencies()` (main.py)
5. Update state availability in component channels if needed

### Adding New Components
1. Create component class with required functionality
2. Implement `Provider` subclass for container registration
3. Register provider in `common_dependencies()`
4. Add to `MossInReachyMini` constructor if needed

### Live Streaming Integration
- Requires `douyin_live` extras installation
- Configuration in `.workspace/configs/douyin_live/douyin_live_config.yaml`
- `LiveAgent` runs in parallel with `MainAgent` when `REACHY_MINI_MODE="live"`
- Barrage classification and event processing in `framework/apps/live/`

### Memory and Session Storage
- Memory: `StorageMemory` for agent context and personality
- Sessions: `StorageSession` for conversation history
- Files stored in `.workspace/runtime/memory/` subdirectories
- UI available via Streamlit for manual inspection/editing

### Testing Strategy
- Unit tests in `tests/framework/` for core components
- Integration tests for agent interactions
- Async tests require `pytest-asyncio`
- Mock hardware dependencies for CI testing

### Version Control
- Branch naming: `feat/` for features, `fix/` for bug fixes, `chore/` for maintenance
- Commit messages: Use conventional prefix (`feat:`, `fix:`, `chore:`, `docs:`, `test:`) followed by concise description
- Chinese descriptions are acceptable for feature descriptions

## Troubleshooting

### Common Issues

1. **Audio device conflicts**: Check `PYAUDIO_INPUT_DEVICE_INDEX` in `.env`
2. **Face recognition failures**: Verify image format and run training script
3. **Live streaming connection**: Validate Douyin credentials and network
4. **State transition errors**: Check state `in_switchable`/`out_switchable` flags

### Debug Mode
- Set `LOG_LEVEL=DEBUG` in `.env` for verbose logging
- Check `.workspace/runtime/logs/` for application logs
- Video recording debug logs in `video_records/.tmp/`

## References

- [MOSShell Framework](https://github.com/GhostInShells/MOSShell)
- [Reachy Mini Documentation](https://docs.reachy.mini/)
- [VolcEngine API Docs](https://www.volcengine.com/docs/)
- [ghoshell_moss](dependencies/ghoshell_moss-0.1.0a0-py3-none-any.whl)