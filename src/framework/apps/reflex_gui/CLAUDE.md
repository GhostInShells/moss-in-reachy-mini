# CLAUDE.md - Reflex GUI

This file provides guidance for working with the Reflex GUI component of MOSS in Reachy Mini.

## Project Overview

The **Reflex GUI** is a modern real-time Web interface for the MOSS in Reachy Mini robot system. It provides a responsive web-based UI that displays robot interactions, conversations, and status updates in real-time through a ZMQ-based streaming architecture.

### Core Components
- **Real-time Markdown Display**: Streams markdown content from MOSS agents to the web interface
- **Dynamic Component System**: Experimental framework for data-driven UI component generation
- **ZMQ Communication**: High-performance messaging between MOSS framework and web app
- **Reflex Framework**: Python full-stack web framework for responsive UI

## Historical Context and Design Discussions

### `.discuss` Directory
The `.discuss` directory contains historical design discussions, analysis reports, and decision documentation for this component. When working on Reflex GUI development, always consult these files to understand the project's evolution and design rationale.

**Key Documents:**
- `reflex_gui_analysis.md`: Comprehensive analysis of the current implementation, technical architecture, and future vision
- *(Additional discussion files may be added over time)*

### How to Use Historical Context
1. **Before making significant changes**: Review relevant `.discuss` documents to understand design decisions
2. **When encountering design questions**: Check if similar issues were previously discussed
3. **For architectural understanding**: Read analysis reports to grasp the overall system design
4. **When planning new features**: Consult existing discussions to ensure consistency with project vision

## Development Setup

### Prerequisites
- Python 3.12+
- [Reflex](https://reflex.dev/) framework (v0.8.27+)
- ZMQ libraries for inter-process communication

### Installation
```bash
# Navigate to the reflex_gui directory
cd src/framework/apps/reflex_gui

# Install Reflex and dependencies
uv pip install reflex==0.8.27
uv pip install pyzmq

# Or use the requirements file
uv pip install -r requirements.txt
```

### Running the Application
```bash
# Start the Reflex development server
reflex run

# The application will be available at http://localhost:3000
```

### Integration with MOSS Framework
The Reflex GUI integrates with the main MOSS system through:
1. **ZMQ Channel**: `tcp://127.0.0.1:9527`
2. **Channel Commands**: `append_markdown`, `clear_markdown`
3. **Main Integration**: See `src/framework/main.py:49-61` for integration details

## Architecture

### Communication Flow
```
MOSS Framework → ZMQ Channel (tcp://127.0.0.1:9527) → Reflex GUI → Browser
```

### Key Files
- `reflex_gui/reflex_gui.py`: Main application with real-time markdown display
- `reflex_gui/stream_gui_test.py`: Experimental dynamic component generation
- `rxconfig.py`: Reflex application configuration
- `.discuss/`: Design discussions and analysis documents

### State Management
- **Async Queue**: `asyncio.Queue` for streaming content between threads
- **Background Events**: `@rx.event(background=True)` for continuous updates
- **Component State**: Reactive state management through Reflex's State class

## Development Guidelines

### Adding New Features
1. **Consult Historical Context**: Review `.discuss` documents for design patterns
2. **Follow Data-Driven Approach**: Design components that can be defined by data structures
3. **Maintain Real-time Capability**: Ensure new features work with the streaming architecture
4. **Test ZMQ Integration**: Verify communication with the main MOSS system

### Component Development
- **Simple Components**: Start with markdown-based content
- **Dynamic Components**: Use the pattern from `stream_gui_test.py` for data-driven UI
- **Styling**: Consider how styles can be dynamically applied (currently experimental)

### Performance Considerations
- **ZMQ Timeouts**: Default receive interval is 3 seconds (optimized for reliability)
- **Async Patterns**: Use `arun_until_closed` for better performance than `run_in_thread`
- **Queue Management**: Ensure proper cleanup of async queues

## Testing

### Manual Testing
```bash
# Test ZMQ communication
python -m src.framework.apps.zmq_test

# Run the Reflex GUI
reflex run
```

### Integration Testing
1. Start the main MOSS application
2. Launch the Reflex GUI
3. Verify real-time content streaming works
4. Test component generation features

## Common Tasks

### Adding New Content Types
1. Define the data structure for the new content
2. Create corresponding UI components in Reflex
3. Update ZMQ channel commands to handle the new type
4. Test with the main MOSS system

### Debugging ZMQ Communication
- Check that the ZMQ server is running on port 9527
- Verify channel names match (`reflex` in both ends)
- Monitor queue sizes and async task status

### Performance Optimization
- Adjust `_receive_interval_seconds` in `ZMQChannelProvider` if needed
- Monitor browser performance with many dynamic components
- Consider pagination or virtualization for large content sets

## Future Development Directions

Based on historical discussions (see `.discuss/reflex_gui_analysis.md`):

### Short-term Goals
1. **Component Library Expansion**: Define more component types (tables, charts, forms)
2. **Style System**: Develop dynamic styling approach
3. **Layout Management**: Handle complex layout relationships

### Long-term Vision
1. **Bidirectional Communication**: Allow GUI to control robot actions
2. **Multi-client Support**: Serve multiple web clients simultaneously
3. **Advanced Visualization**: Real-time sensor data and robot state visualization

## Related Resources

- **Main Project CLAUDE.md**: `/Users/wangshiqi/projects/ghosts/moss-in-reachy-mini/CLAUDE.md`
- **ZMQ Test Script**: `src/framework/apps/zmq_test.py`
- **MOSS Integration**: `src/framework/main.py:49-61`

---

*Remember: Always consult the `.discuss` directory for historical context before making significant architectural changes.*