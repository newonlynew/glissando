from typing import Protocol
from pathlib import Path

from glissando.getter.message import Messages

class BaseGetter(Protocol):
    def get_messages(self, filepath: Path) -> Messages:
        ...
