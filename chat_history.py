
import os
import datetime
import csv
import pandas as pd

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


LOG_FILE = "log_history.csv"

class CustomChatMessageHistory(ChatMessageHistory):
    def __init__(self, msg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.addition(msg)
        self.add_messages(msg)
        
    def addition(self):
        self.add_messages(self.msg)
        

class ChatHistoryHelper:
    def __init__(self) -> None:
        self.log_file_path = LOG_FILE
        self.columns = ["session_id","human_message","ai_response","timestramp"]
        self.create_log_file()
        
    def create_log_file(self):
        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, 'w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.columns)
                # Write column headers
                writer.writeheader()
    
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        df = pd.read_csv("log_history.csv")
        if df.empty:
            session = ChatMessageHistory()
        
        # Use boolean indexing to filter rows based on the search key in the specified column
        filtered_df = df[df["session_id"] == int(session_id)]
        
        # If any row matches the search criteria, return the first matching row
        if not filtered_df.empty:
            # formatted_msg = filtered_df[["human_message","ai_response"]].to_dict(orient="records")
            formatted_msg = [f"human:{row['human_message']}, ai:{row['ai_response']}" for index, row in df.iterrows()]
            session = CustomChatMessageHistory(formatted_msg) #.values
        else:
            session = ChatMessageHistory()
        return session

    def get_processed_chat_history(self, session_history):
        output = session_history.messages
        print("list of chat history : ", output)
        return output
        

    def log_chat_histroy(self, session_id, query, response):
        with open(self.log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([session_id, query, response, datetime.datetime.now()])
