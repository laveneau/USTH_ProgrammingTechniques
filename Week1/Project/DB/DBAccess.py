# This file should not be modified
# Do as if it was automatically generated...
from typing import List, Any, Dict, Type, Union


class DBEntry:
    """
    Entry into the DataBase

    An entry corresponds to a row of a table.
    The scheme of the table is given by the columns names and data type,
    except the first value (the key) which is given apart.
    The values should be sorted according to this scheme...
    """

    def __init__(self, key: int, columns: List[str], values: List[Any]):
        self.__key = key
        self.__columns = columns
        self.__values = {}
        for i in range(len(columns)):
            self.__values[columns[i]] = values[i]

    @property
    def key(self) -> int:
        """
        Returns the key of this database entry.

        :return: key of the entry
        """
        return self.__key

    def get_value_by_column(self, column: str) -> Any:
        """
        Returns the value corresponding to the column name of this database entry.

        :param column: name of the column for which the value should be returned
        :return: value of this entry at the column location
        """
        return self.__values[column]

    def set_value_by_column(self, column: str, value: Any) -> None:
        """
        Sets the value for a given column of this database entry.

        :param column: the column, given by is scheme name
        :param value: the value to set
        :return: None
        """
        self.__values[column] = value


class DBTable:
    """
    Class representing a database table.

    A table is organised with a simplified table scheme as follows:
    - there is a single key of type INT, not represented into the scheme
    - the others columns are given by a name, and no type verification is done

    Actually this is a very simplified table model...
    """

    def __init__(self, name: str, columns: List[str], data_types: List[Type]):
        """
        Creates a new instance of DBTable given the table name, and its scheme (columns names and datatype)

        Notice that the datatype should be a list of datatype.
        A example of use if the following:
        ```python
        names = [ 'first_name', 'name', 'phone', 'date_of_birth', 'book_limit' ]
        data_types = [ str, str, str, datetime.date, int ]
        table = DBTable('Users', names, data_types)
        ```
        :param name: Name of the table
        :param columns: Scheme of the table (name of its columns)
        :param data_types: Scheme of the table (datatype of its columns)
        """
        self.__name = name
        self.__columns = columns
        self.__nb_columns = len(columns)
        self.__data: Dict[int, DBEntry] = {}
        self.__data_types = data_types

    def entry_exists(self, key: int) -> bool:
        """
        Predicates returning True if an entry exists with the key in the table, False else.

        :param key: The key to search in the table
        :return: True iff there is an entry with the corresponding key.
        """
        for entry in self.__data:
            if entry == key:
                return True
        return False

    def get_entry(self, key: int) -> DBEntry:
        """
        Gets an entry for a given key.

        Notice that if the key does not correspond to an entry of the table, the exception `Exception` is raised...
        :param key: Key of the searched entry
        :return: The entry with the given key
        """
        if not self.entry_exists(key):
            raise Exception(f'entry {key} does not exists')

        return self.__data[key]

    def add_entry(self, key: int, values: List[Any]) -> bool:
        """
        Adds an entry to this table.

        If there already exists an entry with the same key, this method return False.
        Else it returns True.

        Notice that an exception is raised (type `Exception`) if the number of values is not good,
        or if the datatype of the values are not good.

        :param key: The key of the element to add
        :param values: The entry to add, given as a list of values corresponding to the scheme of the table
        :return: True if the entry is added, False else (key already used)
        """
        if self.entry_exists(key):
            return False

        nb_columns = len(values)
        if nb_columns != self.__nb_columns:
            raise Exception(f'bad number of columns: {nb_columns}/{self.__nb_columns}')
        for i in range(nb_columns):
            expected: Type = self.__data_types[i]
            if not isinstance(values[i], expected):  # noqa
                raise Exception(f'bad datatype for value number {i}: expect {expected}, got {type(values[i])}')

        self.__data[key] = DBEntry(key, self.__columns, values)

        return True

    def generate_key(self):
        """
        Returns an unused key that can be used to add new value...

        :return: A ready to use key
        """
        if not self.__data:
            return 0
        return max(self.__data) + 1


class DBAccess:
    """
    This module stores a database.

    There is only one database, stored into the class.
    Notice that it is not necessary to create any instance, all the data and methods are stored into the class!

    The module allows to create new table, and to retrieve them (the created table are stored).
    The data are not stored permanently, and will be dropped after the program close...
    """
    __tables: Dict[str, DBTable] = {}

    @classmethod
    def add_table(cls, name: str, description: List[str], data_types: List[Any]) -> DBTable:
        """
        Adds a new table into the database.

        Notice that the name of the table cannot be used more than once.
        Hence, if you add twice a table with the same name, you will obtain an Exception the second time.

        :param name: Name of the table to add.
        :param description: Scheme of the table, given as the names of each column.
        :param data_types: Scheme of the table, given as the data type of each column.
        :return: The created table, that was stored into the DB.
        """
        if name in cls.__tables:
            raise Exception(f'table {name} already exists')

        table = DBTable(name, description, data_types)
        cls.__tables[name] = table

        return table

    @classmethod
    def is_table(cls, name: str) -> bool:
        """
        Predicates returning if a table (known by its name) is already stored into the DB.

        :param name: Name of the table to search in.
        :return: True if the table exists in the DB, False else.
        """
        return name in cls.__tables

    @classmethod
    def get_table(cls, name: str) -> Union[DBTable, None]:
        """
        Gets a Table from its name, if stored into the DB.
        Else returns None.

        :param name: Name of the Table to return.
        :return: The DBTable if it is stored into the DB, None else.
        """
        if cls.is_table(name):
            return cls.__tables[name]
        return None
