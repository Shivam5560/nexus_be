from pymongo import MongoClient
import logging

# Global variables to store MongoDB client and database
mongo_client = None
db = None


def initialize_db(app):
    """Initialize MongoDB connection."""
    global mongo_client, db
    try:
        mongo_client = MongoClient(app.config["MONGO_URI"])
        # Explicitly specify database name
        db_name = app.config.get(
            "MONGO_DB_NAME", "nexus_db"
        )  # Default to 'nexus_db' if not specified
        db = mongo_client[db_name]
        logging.info(
            f"MongoDB connection established successfully to database: {db_name}"
        )
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise


def get_db():
    """Get database instance."""
    global db
    if db is None:
        raise RuntimeError("Database not initialized. Call initialize_db first.")
    return db


def close_db_connection():
    """Close MongoDB connection."""
    global mongo_client
    if mongo_client:
        mongo_client.close()
