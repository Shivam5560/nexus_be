from flask import Flask
from app.config import config
from app.services.db_service import initialize_db
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

bcrypt = Bcrypt()
jwt = JWTManager()

def create_app(config_name="default"):
    """Application factory function."""
    app = Flask(__name__)

    from app.config import config
    app.config.from_object(config[config_name])

    bcrypt.init_app(app)
    jwt.init_app(app)

    initialize_db(app)

    from app.routes.user_route import register_auth_routes
    register_auth_routes(app)

    from app.routes.resume_route import register_resume_routes
    register_resume_routes(app)

    return app
