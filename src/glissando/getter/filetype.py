from enum import Enum, unique


@unique
class FileType(Enum):
    STANDART = "standart"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
