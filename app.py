from langChainBot import answer_question
from flask import Flask,request,make_response,jsonify
app = Flask(__name__)


@app.route('/lang-chain',methods=['POST'])
def lang_chain(current_user):
    try:
        data = request.json
        query = data.get('query')
        files = data.get('files',[])
        query_answer = answer_question(query,files)
        return make_response(jsonify({"status": "Success", "message": "Query result fetch successfully.", "data": query_answer}), 200) 
    except Exception as e:
       return make_response(jsonify({"status": "Error", "message": str(e), "data": ""}), 500)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)