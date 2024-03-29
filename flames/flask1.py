# https://www.geeksforgeeks.org/flask-creating-first-simple-application/

from flask import Flask, redirect, url_for, request

app = Flask(__name__)

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        print('Got message from POST')
        return redirect(url_for('success',name = user))
    else:
        user = request.args.get('nm')
        print('Didn\'t get message from Post')
        return redirect(url_for('success',name = user))

if __name__ == '__main__':
    app.run(debug = True)
