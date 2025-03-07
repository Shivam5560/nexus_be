import os
from flask import Flask

def create_app(config_name):
    """
    Flask application factory that creates and configures the app.
    """
    # Create the Flask app
    app = Flask(__name__)
    
    # Load configuration based on environment
    from app.config import config_by_name
    app.config.from_object(config_by_name[config_name])
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from app.controllers.grammar_controller import grammar_bp
    from app.controllers.resume_controller import resume_bp
    
    app.register_blueprint(grammar_bp, url_prefix='/api/grammar')
    app.register_blueprint(resume_bp, url_prefix='/api/resume')
    
    # Register a health check route
    @app.route('/health')
    def health():
        from app.utils.healthUtils import check_health
        status = check_health()
        return {"status": status}
    
    return app