from database import r
import bcrypt

def create_user(username, password, verify_password, mobile, firstname, lastname):
    # Check if passwords match
    if password != verify_password:
        return "Passwords do not match"
    
    # Check if the username already exists
    if r.exists(f"user:{username}"):
        return "Username already exists"
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Store the user information in the database
    r.hset(f"user:{username}", mapping={
        "username": username,
        "password": hashed_password.decode('utf-8'),
        "mobile": mobile,
        "firstname": firstname,
        "lastname": lastname
    })
    
    return "User created successfully"

def authenticate(username, password):
    # Retrieve the user data from the database
    user_data = r.hgetall(f"user:{username}")
    
    # Check if the user exists
    if not user_data:
        return False
    
    # Decode the stored password and compare it with the provided password
    stored_password = user_data.get('password')
    if stored_password is None:
        return False
    
    return bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8'))

