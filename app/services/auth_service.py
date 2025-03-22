from app.models.user_model import User
from app.services.db_service import get_db
from app import bcrypt
from bson import ObjectId
from pymongo.errors import DuplicateKeyError


def create_user(username, email, password, role="user"):
    db = get_db()

    if db.users.find_one({"$or": [{"username": username}, {"email": email}]}):
        return None, "Username or email already exists"

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    user = User(username=username, email=email, password=hashed_password, role=role)

    try:
        result = db.users.insert_one(user.to_dict())
        user._id = result.inserted_id
        return user, None
    except DuplicateKeyError:
        return None, "Username or email already exists"
    except Exception as e:
        return None, str(e)


def get_user_by_id(user_id):
    db = get_db()
    obj_id = ObjectId(user_id)
    user_data = db.users.find_one({"_id": obj_id})
    if user_data:
        return User.from_dict(user_data)
    return None


def get_user_by_username(username):
    db = get_db()
    user_data = db.users.find_one({"username": username})
    if user_data:
        return User.from_dict(user_data)
    return None


def get_user_by_email(email):
    db = get_db()
    user_data = db.users.find_one({"email": email})
    if user_data:
        return User.from_dict(user_data)
    return None


def validate_user(username_or_email, password):
    user = get_user_by_username(username_or_email) or get_user_by_email(
        username_or_email
    )

    if not user:
        return None, "Invalid username or email"

    if not bcrypt.check_password_hash(user.password, password):
        return None, "Invalid password"

    return user, None


def init_auth_db():
    db = get_db()

    db.users.create_index("username", unique=True)
    db.users.create_index("email", unique=True)
