from flask import request, jsonify
from dependency_injector.wiring import inject, Provide
from ._routes import add_endpoint, url_routes
from celery import current_app as app
from celery.result import AsyncResult
import celery.states as states

def index():
    # Return an endpoint:rule list
    result = { u['endpoint']: u['rule'] for u in url_routes }
    return jsonify(result)

def task():
    a= 10
    b = 20
    rq = request
    celery = app
    print("HELLO: {} {} {} {}".format(a,b,celery,rq))
    task = celery.send_task('hello', args=['World'])
    print('Hello task submitted')
    if task.status != 'SUCCESS':
        return jsonify(task.id)
    return jsonify(task.result)

def check():
    res = app.AsyncResult(request.args.get('id'))
    return res.state if res.state == states.PENDING else str(res.result)

add_endpoint('/', 'index', index, methods=['GET'])
add_endpoint('/task', 'task', task, methods=['GET'])
add_endpoint('/check', 'check', check, methods=['GET'])