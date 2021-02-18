from celery import current_app
import celery.states as states


class TaskService:
    def create_task(self, name, args):
        task = current_app.send_task(name, args=args)
        if task.status != 'SUCCESS':
            return {'task': task.id}
        return task.result

    def get_result(self, id):
        res = current_app.AsyncResult(id)
        return res.state if res.state == states.PENDING else res.result