from pathlib import Path

from .adapters import (
    BaseGetter,
    StandartGetter,
    TelegramGetter,
)
from .message import Messages
from .filetype import FileType

getters = {
    FileType.STANDART: StandartGetter,
    FileType.TELEGRAM: TelegramGetter,
}


class MessagesGetter(BaseGetter):
    def __init__(self, filetype: FileType) -> None:
        self._getter: BaseGetter = getters[filetype]()

    @classmethod
    def from_filetype(cls, filetype: FileType) -> "MessagesGetter":
        return cls(filetype)

    def get_messages(self, filepath: Path) -> "Messages":
        messages = self._getter.get_messages(filepath)
        return messages
