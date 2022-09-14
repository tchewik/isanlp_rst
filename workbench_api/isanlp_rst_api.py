from flask import Flask, make_response
from flask import request
import os
import sys
import json

from parser_wrapper import Parser

app = Flask(__name__)


@app.route('/parse', methods=['POST', 'GET'])
def request_handler():
    print('request.headers', request.headers['Content-Type'], file=sys.stderr)

    if request.headers['Content-Type'].startswith('application/json'):
        session = request.get_json(force=True)
    else:
        session = request.form
    text = session.get('input')

    if text:
        results = parser(text)
        response = make_response(results)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    return make_response("Something went wrong.")


if __name__ == "__main__":
    # Load BERT #####
    parser = Parser(syntax_address=("", 3331), rst_address=("", 3332))

    # Run app #####
    port = int(os.environ.get("PORT", 9105))
    app.run(host='0.0.0.0', port=port)
