import uuid
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from cryptography.fernet import Fernet
import env as en

class Auth:
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or en.redis_client
        self.ph = PasswordHasher()
        self.cipher_suite = Fernet(Fernet.generate_key())

    def register(self, username, password, name, email, contact_number):
        if self.redis_client.exists(f"user:{username}"):
            return False, "Username already exists"
        
        user_id = str(uuid.uuid4())
        hashed_password = self.ph.hash(password)
        user_data = {
            "user_id": user_id,
            "username": username,
            "password": hashed_password,
            "name": self.encrypt(name),
            "email": self.encrypt(email),
            "contact_number": self.encrypt(contact_number)
        }
        self.redis_client.hmset(f"user:{username}", user_data)
        return True, "Registration successful"

    def login(self, username, password):
        print(f"Username: {username}, Password: {password}")  # For debugging
        user_data = self.redis_client.hgetall(f"user:{username}")
        if not user_data:
            return False, "User not found", None, None
        
        user_dict = {k.decode(): v.decode() for k, v in user_data.items()}
        
        try:
            self.ph.verify(user_dict['password'], password)
            session_id = str(uuid.uuid4())
            self.redis_client.setex(f"session:{session_id}", 3600, user_dict['user_id'])
            return True, "Login successful", session_id, user_dict['user_id']
        except VerifyMismatchError:
            return False, "Incorrect password", None, None

    def logout(self, session_id):
        self.redis_client.delete(f"session:{session_id}")

    def get_user_details(self, user_id):
        user_data = self.redis_client.hgetall(f"user:{user_id}")
        if not user_data:
            return None
        
        user_dict = {k.decode(): v.decode() for k, v in user_data.items()}
        return {
            "user_id": user_dict["user_id"],
            "username": user_dict["username"],
            "name": self.decrypt(user_dict["name"]),
            "email": self.decrypt(user_dict["email"]),
            "contact_number": self.decrypt(user_dict["contact_number"])
        }

    def validate_session(self, session_id, user_id):
        stored_user_id = self.redis_client.get(f"session:{session_id}")
        return stored_user_id and stored_user_id.decode() == user_id

    def encrypt(self, data):
        return self.cipher_suite.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()