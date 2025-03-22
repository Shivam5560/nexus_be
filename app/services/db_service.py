from pymongo import MongoClient
import logging


mongo_client = None
db = None


def initialize_db(app):

    global mongo_client, db
    try:
        mongo_client = MongoClient(app.config["MONGO_URI"])

        db_name = app.config.get("MONGO_DB_NAME", "nexus_db")
        db = mongo_client[db_name]
        logging.info(
            f"MongoDB connection established successfully to database: {db_name}"
        )
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        raise


def get_db():

    global db
    if db is None:
        raise RuntimeError("Database not initialized. Call initialize_db first.")
    return db


def close_db_connection():

    global mongo_client
    if mongo_client:
        mongo_client.close()
