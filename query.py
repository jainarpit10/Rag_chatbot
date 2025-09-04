from langChainBot import answer_question
import os

current_dir = os.path.dirname(__file__)   # folder where this script is
pdf_path = os.path.join(current_dir, "accenture.pdf")

def lang_chain():
    try:
        query = input(str("Ask your query: "))
        files = pdf_path
        query_answer = answer_question(query,files)
        return  query_answer
    except Exception as e:
       return  str(e)
    
result  = lang_chain()
print(f"The result of your query is {result}.")