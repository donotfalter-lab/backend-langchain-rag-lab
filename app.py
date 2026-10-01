"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""

        payload = request.get_json(silent=True)
        question, error = validate_question_payload(payload)
        if error:
            return jsonify(error), 400

        try:
            response = answer_question(question)
        except LangChainServiceError as service_error:
            return (
                jsonify(
                    format_error_response(
                        "langchain_service_error", str(service_error)
                    )
                ),
                502,
            )

        return jsonify(response), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
