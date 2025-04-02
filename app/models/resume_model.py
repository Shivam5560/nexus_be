from datetime import datetime
from bson import ObjectId
from app.utils.resume_util import generate_unique_id

class Resume:
    def __init__(self, file_path, user_id, resume_id=None, status=None, uploaded_at=None, _id=None):
        self._id = _id if _id else ObjectId()
        self.file_path = file_path
        self.user_id = ObjectId(user_id)
        self.resume_id = generate_unique_id() if resume_id is None else resume_id
        self.status = status if status else "ACTIVE"
        self.uploaded_at = uploaded_at if uploaded_at else datetime.now()

    @staticmethod
    def from_dict(resume_dict):
        return Resume(
            file_path=resume_dict.get('file_path'),
            user_id=resume_dict.get('user_id'),
            resume_id=resume_dict.get('resume_id'),
            status = resume_dict.get('status'),
            uploaded_at=resume_dict.get('uploaded_at'),
            _id=resume_dict.get('_id')
        )

    def to_dict(self):
        return {
            '_id': self._id,
            'file_path': self.file_path,
            'resume_id': self.resume_id,
            'user_id': self.user_id,
            'status': self.status,
            'uploaded_at': self.uploaded_at
        }

    def to_json(self):
        resume_dict = self.to_dict()
        resume_dict['_id'] = str(resume_dict['_id'])
        resume_dict['user_id'] = str(resume_dict['user_id'])
        resume_dict['resume_id'] = str(resume_dict['resume_id'])
        resume_dict['status'] = str(resume_dict['status'])
        resume_dict['uploaded_at'] = resume_dict['uploaded_at'].isoformat()
        return resume_dict