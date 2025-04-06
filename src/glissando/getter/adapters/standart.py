import json
from pathlib import Path

from .base import BaseGetter
from glissando.getter.message import Messages, Message


class StandartGetter(BaseGetter):
    def get_messages(self, filepath: Path) -> Messages:
        messages = []
        raw_messages = load_json(filepath)
        for raw_message in raw_messages:
            message = Message(
                author=raw_message["author"],
                text=raw_message["text"],
            )
            messages.append(message)
        return Messages(messages)


def load_json(filepath: Path) -> dict:
    with open(filepath) as file:
        return json.load(file)
