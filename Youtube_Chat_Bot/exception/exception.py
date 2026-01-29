import os
import sys
from Youtube_Chat_Bot.logging import logger

class YoutubeChatBotException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message
        _,_,exc_info=error_details.exc_info()
        
        self.lineno=exc_info.tb_lineno
        self.file_name=exc_info.tb_frame.f_code.co_filename
        
    def __str__(self):
        return f"Error occured in python script name [{self.file_name}] line number [{self.lineno}] error message [{self.error_message}]"
    