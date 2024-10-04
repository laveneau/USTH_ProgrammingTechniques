# DO NOT MODIFY THIS CLASS
import datetime

from DB import DBAccess
from .tools import static_initializer


@static_initializer
class UserDataAccess:
    @classmethod
    def static_init(cls):
        cls.__table_name = 'users'
        cls.__table_description = ['client_id', 'date_of_birth', 'email_address', 'first name', 'surname',
                                   'has_credit_limit',
                                   'credit_limit']
        cls.__table_data_types = [int, datetime.date, str, str, str, bool, int]
        cls.__users = DBAccess.add_table(cls.__table_name, cls.__table_description, cls.__table_data_types)

    @classmethod
    def add_user(cls, user):
        table = DBAccess.get_table(cls.__table_name)
        key = table.generate_key()
        values = [
            user.client.id,
            user.date_of_birth,
            user.email_address,
            user.first_name,
            user.surname,
            user.has_credit_limit,
            user.credit_limit,
        ]
        table.add_entry(key, values)
