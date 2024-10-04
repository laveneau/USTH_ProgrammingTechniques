class User:

    def __init__(self, client, date_of_birth, email_address, first_name, surname):
        self.client = client
        self.date_of_birth = date_of_birth
        self.email_address = email_address
        self.first_name = first_name
        self.surname = surname
        self.has_credit_limit = None
        self.credit_limit = -1


