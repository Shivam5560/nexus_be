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

    # Initialize MongoDB
    initialize_db(app)

    # Register blueprints
    from app.routes.appHealthRoutes import health_bp
    app.register_blueprint(health_bp)

    from app.routes.userRoutes import auth_bp
    app.register_blueprint(auth_bp)

    return app
