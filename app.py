
from pathlib import Path
from flask import Flask
from app.routes import main

PROJECT_ROOT = Path(__file__).resolve().parent

def create_app():
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "app" / "templates"),
        static_folder=str(PROJECT_ROOT / "app" / "static")
    )

    app.register_blueprint(main)

    return app


if __name__ == "__main__":
    application = create_app()

    print("Template directory:", application.template_folder)
    print("Template exists:", application.jinja_loader.get_source(
        application.jinja_env, "index.html"
    )[1])

    application.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
