import os 
import chromadb

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# module imports
from load_embeddings import LoadEmbedding
from chat_history import ChatHistoryHelper
from enums import Prompt
from reranking import rerank


class Executor:
    def get_top_k_docs(self, query):
        top_k = 5
        
        client = chromadb.HttpClient(host="localhost", port=8000)

        # Get the stored vector db
        embedding = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        vectordb = Chroma(
            client=client,
            embedding_function=embedding
        )

        relevant_docs = vectordb.max_marginal_relevance_search(query,
                                                            k=8,
                                                            #    filter={"source": INPUT_FILE_PATH}
                                                            )
        return rerank(query=query, relevant_docs=relevant_docs, top_k=top_k)


    def get_contextualized_qa_chain(self):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

        contextualize_q_system_prompt = Prompt.contextualize_q_system_prompt.value

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        return contextualize_q_chain


    def get_contextualized_question(self, chat_history, query):
        if chat_history:
            contextualized_qa_chain = self.get_contextualized_qa_chain()
            contextualized_question = contextualized_qa_chain.invoke({
                "chat_history": chat_history,
                "question": query
            })
            print("here is the context question..........")
            print(contextualized_question)
            return contextualized_question
        else:
            return query


    def question_answer(self, session_id, query):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

        qa_system_prompt = Prompt.qa_system_prompt.value

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        rag_chain = (
                qa_prompt | llm | StrOutputParser()
        )

        session_history = ChatHistoryHelper().get_session_history(session_id)
        chat_history = ChatHistoryHelper().get_processed_chat_history(session_history)
        context_query = self.get_contextualized_question(chat_history, query)
        context = self.get_top_k_docs(query=context_query)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            ChatHistoryHelper().get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        response = conversational_rag_chain.invoke(
            {
                "question": query,
                "chat_history": chat_history,
                "context": context
            },
            config={
                "configurable": {"session_id": session_id}
            },  # constructs a key "abc123" in `store`.
        )
        print("\nAnswer:\n\n")
        print(response)
        ChatHistoryHelper().log_chat_histroy(session_id, query, response)

        return response




if __name__=="__main__":
    db = LoadEmbedding().execute()
        
    query = input("ask your query:")
    session_id = input("session id:")
    Executor().question_answer(session_id, query)
    
    


