from flask import Flask
from flask import request,make_response,render_template
import urllib.parse
from use_model import classify
app = Flask(__name__)
@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/classify',methods=['POST'])
def say_hello():
  URL = request.form.get('url')
  print({'url':URL})
  CLASS = classify(URL)
  data = ''
  if(CLASS!=None):
    data= '{"success":true, "class_name" : "'+CLASS+'"}'
  else:
    data = '''{
      "success":false,
      "message": "Invalid URL"
    }'''
  r = make_response( data )
  r.mimetype = 'application/json'
  return r
app.run(host='0.0.0.0', port=81,debug=True)
