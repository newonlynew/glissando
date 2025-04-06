from typing import List, Iterator
from dataclasses import dataclass


@dataclass
class Message:
    author: str
    text: str


@dataclass
class Messages:
    messages: List[Message]

    def to_list(self) -> List[str]:
        return [message.text for message in self.messages]

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)
