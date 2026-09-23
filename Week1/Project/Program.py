# DO NOT MODIFY THIS FILE AT ALL!
import datetime
from sys import argv

from LegacyApp import UserService


class Program:

    @staticmethod
    def prove_add_user(args):
        user_service = UserService()
        add_result = user_service.add_user("Bob", "Sponge", "spongebob@spongemail.com", datetime.date(1969, 12, 4), 3)
        message = "successful" if add_result else "unsuccessful"
        print(f'Adding spongebob was {message}')

    @staticmethod
    def main(args):
        Program.prove_add_user(args)


if __name__ == "__main__":
    Program.main(argv)
