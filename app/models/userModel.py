from datetime import datetime
from bson import ObjectId


class User:
    def __init__(self, username, email, password, role='user', created_at=None, _id=None):
        self._id = _id if _id else ObjectId()
        self.username = username
        self.email = email
        self.password = password
        self.role = role
        self.created_at = created_at if created_at else datetime.now()

    @staticmethod
    def from_dict(user_dict):
        return User(
            username=user_dict.get('username'),
            email=user_dict.get('email'),
            password=user_dict.get('password'),
            role=user_dict.get('role', 'user'),
            created_at=user_dict.get('created_at'),
            _id=user_dict.get('_id')
        )

    def to_dict(self):
        return {
            '_id': self._id,
            'username': self.username,
            'email': self.email,
            'password': self.password,
            'role': self.role,
            'created_at': self.created_at
        }

    def to_json(self):
        user_dict = self.to_dict()
        user_dict['_id'] = str(user_dict['_id'])  # Convert ObjectId to string
        user_dict['created_at'] = user_dict['created_at'].isoformat()  # Convert datetime to string
        user_dict.pop('password', None)  # Remove password
        return user_dict