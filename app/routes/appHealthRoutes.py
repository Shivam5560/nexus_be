from flask import Blueprint, jsonify
from app.controllers.appHealthController import check_health

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    health_data = check_health()

    # Set appropriate HTTP status code based on health status
    status_code = 200 if health_data["status"] == "UP" else 503

    return jsonify(health_data), status_code
