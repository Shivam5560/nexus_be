import os
from app import create_app
from app.utils.health_util import check_health

app = create_app(os.getenv("FLASK_ENV", "production"))

health_status = check_health()
print(f"Health status: {health_status}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)),threaded=True)
