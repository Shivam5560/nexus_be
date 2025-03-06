from flask import Flask
from app.config import config
from app.services.dbServices import initialize_db
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

    from app.routes.authRoutes import register_auth_routes
    register_auth_routes(app)

    from app.routes.fileRoutes import register_file_routes
    register_file_routes(app)

    return app
