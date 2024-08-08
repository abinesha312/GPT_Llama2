from database import r
import bcrypt

def create_user(username, password, verify_password, mobile, firstname, lastname):
    if password != verify_password:
        return "Passwords do not match"
    if r.exists(f"user:{username}"):
        return "Username already exists"
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    r.hset(f"user:{username}", mapping={
        "username": username,
        "password": hashed_password.decode('utf-8'),
        "mobile": mobile,
        "firstname": firstname,
        "lastname": lastname
    })
    
    return "User created successfully"

def authenticate(username, password):
    user_data = r.hgetall(f"user:{username}")
    
    if not user_data:
        return False
    
    stored_password = user_data['password'].encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), stored_password)