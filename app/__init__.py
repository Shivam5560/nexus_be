from flask import Flask
from app.config import config
from app.services.dbServices import initialize_db


def create_app(config_name="default"):
    """Application factory function."""
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize MongoDB
    initialize_db(app)

    # Register blueprints
    from app.routes.appHealthRoutes import health_bp

    app.register_blueprint(health_bp)

    return app
