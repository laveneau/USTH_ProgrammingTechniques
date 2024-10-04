from enum import Enum, auto


class ClientStatus(Enum):
    VIP = auto(),
    IP = auto(),
    NORMAL = auto(),
    UNDEFINED = auto()
