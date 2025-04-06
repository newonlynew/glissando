import json
from pathlib import Path

from .base import BaseGetter
from glissando.getter.message import Messages, Message


class TelegramGetter(BaseGetter):
    def get_messages(self, filepath: Path) -> Messages:
        messages = []
        data = load_json(filepath)
        raw_messages = data["messages"]
        for raw_message in raw_messages:
            author = raw_message.get("from")
            text = raw_message.get("text")
            if not isinstance(text, str):
                continue
            if not isinstance(author, str):
                continue
            message = Message(
                author=raw_message["from"],
                text=raw_message["text"],
            )
            messages.append(message)
        return Messages(messages)


def load_json(filepath: Path) -> dict:
    with open(filepath) as file:
        return json.load(file)
