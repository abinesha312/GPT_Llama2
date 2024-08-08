import redis
import bcrypt

# Connect to Redis
r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

def create_user(username, password):
    # Check if username already exists
    if r.exists(f"user:{username}"):
        print("Username already exists.")
        return False
    
    # Hash the password
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Store user info in a hash
    r.hset(f"user:{username}", mapping={
        "username": username,
        "password": hashed_password.decode('utf-8')
    })
    
    print(f"User {username} created successfully.")
    return True

# Example usage
username = input("Enter new username: ")
password = input("Enter new password: ")
create_user(username, password)
