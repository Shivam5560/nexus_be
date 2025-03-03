from flask import jsonify, request
from flask_jwt_extended import create_access_token
from app.services.authServices import create_user, validate_user


def register():
    """
    Register a new user
    """
    data = request.get_json()

    # Extract user data
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    # Validate data
    if not all([username, email, password]):
        return jsonify({"msg": "Missing required fields"}), 400

    # Create user
    user, error = create_user(username, email, password)
    if error:
        return jsonify({"msg": error}), 400

    # Return success response
    return jsonify({
        "msg": "User registered successfully",
        "user": user.to_json()
    }), 201


def login():
    """
    Authenticate a user and return a JWT
    """
    data = request.get_json()

    # Extract login data
    username_or_email = data.get('username') or data.get('email')
    password = data.get('password')

    # Validate data
    if not all([username_or_email, password]):
        return jsonify({"msg": "Missing username/email or password"}), 400

    # Validate user
    user, error = validate_user(username_or_email, password)
    if error:
        return jsonify({"msg": error}), 401

    # Create access token
    access_token = create_access_token(identity=str(user._id))

    # Return token
    return jsonify({
        "msg": "Login successful",
        "access_token": access_token,
        "user": user.to_json()
    }), 200