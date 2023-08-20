from flask import Flask
from src.logger import logging
from src.exception import CustomException
import os, sys


app =Flask(__name__)
@app.route('/',methods=['GET','POST'])
def index():

    try:
        raise Exception("we are testing our exception file")
    except Exception as e:
        ML=CustomException(e, sys)
        logging.info(ML.error_message)
        logging.info("we are testing our logging file")

        return "Welcome to My new Project"

if __name__ == '__main__':
    app.run(debug=True)