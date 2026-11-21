from dataclasses import dataclass

@dataclass
class AppState:
    version: str = "0.1.0"
    debug: bool = None
    verbose: bool = None
    depth: int = None

state = AppState()