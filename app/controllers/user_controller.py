from flask import jsonify, request
from flask_jwt_extended import create_access_token
from app.services.auth_service import create_user, validate_user


def register():
    data = request.get_json()

    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not all([username, email, password]):
        return jsonify({"msg": "Missing required fields"}), 400

    user, error = create_user(username, email, password)
    if error:
        return jsonify({"msg": error}), 400

    return jsonify({"msg": "User registered successfully", "user": user.to_json()}), 201


def login():
    data = request.get_json()

    username_or_email = data.get("username") or data.get("email")
    password = data.get("password")

    if not all([username_or_email, password]):
        return jsonify({"msg": "Missing username/email or password"}), 400

    user, error = validate_user(username_or_email, password)
    if error:
        return jsonify({"msg": error}), 401

    access_token = create_access_token(identity=str(user._id))

    return (
        jsonify(
            {
                "msg": "Login successful",
                "access_token": access_token,
                "user": user.to_json(),
            }
        ),
        200,
    )
