# Flask app factory
import logging

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    
    return app
