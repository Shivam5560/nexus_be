import time
from app.services.dbServices import get_db


def check_health():
    """
    Check application health including MongoDB connection.
    Returns a dictionary with health status information.
    """
    start_time = time.time()

    # Check database connection
    db_status = "UP"
    db_error = None
    try:
        # Simple ping to check MongoDB connection
        db = get_db()
        db.command("ping")
    except Exception as e:
        db_status = "DOWN"
        db_error = str(e)

    response_time = time.time() - start_time

    health_data = {
        "status": "UP" if db_status == "UP" else "DOWN",
        "timestamp": time.time(),
        "components": {"database": {"status": db_status, "error": db_error}},
        "responseTime": f"{response_time:.4f}s",
    }

    return health_data
