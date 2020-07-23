from flask import Flask
from flask import request,make_response
import urllib.parse
from use_model import classify
app = Flask(__name__)
@app.route('/')
def index():
  return 'Server Workssss!'
  
@app.route('/classify',methods=['POST'])
def say_hello():
  URL = request.form.get('URL')
  print(URL)
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