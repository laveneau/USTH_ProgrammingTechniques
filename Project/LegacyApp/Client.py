from .ClientStatus import ClientStatus


class Client:
    def __init__(self, id=-1, name='', status=ClientStatus.UNDEFINED):
        self.__id = -1
        self.__name = ''
        self.__status = status

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def status(self):
        return self.__status

    @status.setter
    def status(self, status):
        self.__status = status
