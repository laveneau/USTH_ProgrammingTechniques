from datetime import datetime

from .ClientStatus import ClientStatus
from .ClientRepository import ClientRepository
from .User import User
from .UserCreditService import UserCreditServiceClient
from .UserDataAccess import UserDataAccess


class UserService:

    # THIS METHOD SHOULD STAY STATIC, with same prototype...
    @staticmethod
    def add_user(firname, surname, email, dateOfBirth, clientId):
        # but you may add typing and you should modify its implementation...
        if (firname == '') or (surname == ''):
            return False
        if ('@' in email) and ('.' not in email):
            return False
        now = datetime.now()
        age = now.year - dateOfBirth.year
        if (now.month < dateOfBirth.month) or ((now.month == dateOfBirth.month) and (now.day < dateOfBirth.day)):
            age = age - 1
        if age < 21:
            return False

        client_repository = ClientRepository()
        client = client_repository.get_by_id(clientId)

        user = User(client=client, date_of_birth=dateOfBirth, email_address=email, first_name = firname, surname=surname)

        if client.status == ClientStatus.VIP:
            user.has_credit_limit = False
        elif client.status == ClientStatus.IP:
            user.has_credit_limit = True
            with UserCreditServiceClient() as user_credit_service:
                credit_limit = user_credit_service.get_credit_limit(user.first_name, user.surname, user.date_of_birth)
                credit_limit = credit_limit * 2
                user.credit_limit = credit_limit
        else:
            user.has_credit_limit = True
            with UserCreditServiceClient() as user_credit_service:
                credit_limit = user_credit_service.get_credit_limit(user.first_name, user.surname, user.date_of_birth)
                user.credit_limit = credit_limit
        
        if user.has_credit_limit and (user.credit_limit < 500):
            return False
        
        UserDataAccess.add_user(user)
        
        return True
