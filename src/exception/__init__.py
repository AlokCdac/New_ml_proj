import os, sys

class CustomException(Exception):
    def __init__(self,error_message:Exception,error_details: sys):
        self.error_message=CustomException.get_detailed_error_message(error_message=error_message, error_details=error_details)
        

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_details:sys)-> str:
        _,_, exce_tb=error_details.exc_info()

        exception_block_line_number=exce_tb.tb_frame.f_lineno
        try_block_line_number =exce_tb.tb_lineno
        file_name=exce_tb.tb_frame.f_code.co_filename

        error_message=f""""
        error occured in executionof :
        [{file_name}] at
        try block line number : [{try_block_line_number}]
        error_message:[{error_message}]
        """
        return error_message
    

    def __str__(self):
        return self.error_message
    
    def __repr___(self):
        return CustomException.__name__.str()
