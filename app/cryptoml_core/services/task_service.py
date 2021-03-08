from celery import current_app, states, current_task
from cryptoml_core.repositories.task_repository import TaskRepository, Task
from cryptoml_core.util.timestamp import get_timestamp
from typing import List
import logging


@current_app.task(name='success_callback')
def on_task_success(result: dict, task: dict):
    logging.info("Task completed: {} TASK: {}".format(on_task_success.name, task))
    service = TaskService()
    try:
        service.store_result(task['id'], result)

    except Exception as e:
        logging.exception(e)
    return result


@current_app.task(name='failure_callback')
def on_task_failure(request, exc, traceback):
    logging.debug(request)
    logging.exception(exc)
    logging.error(traceback)
    pass


class TaskService:
    def __init__(self):
        self.repo = TaskRepository()

    def send(self, task_name, task_args: dict, **kwargs):
        task = Task(
            task_name=task_name,
            args=task_args,
            name=kwargs.get('name'),
            batch=kwargs.get('batch'),
            status='CREATED'
        )
        task = self.repo.create(task)
        celery_task = current_app.send_task(task_name,
                                            args=[task_args],
                                            task_id=task.id,
                                            link=on_task_success.s(task.dict()),
                                            link_error=on_task_failure.s(),
                                            countdown=kwargs.get('countdown')
                                            )
        celery_task.forget()  # Results are saved on task success!
        return task

    def refresh_task(self, task):
        res = current_app.AsyncResult(task.id)
        if res.state != task.status:
            task.status = res.state
            self.repo.update(task.id, task)

    def get_task(self, task_id):
        task = self.repo.get(task_id)
        if task.status not in states.READY_STATES:
            self.refresh_task(task)
        return task

    def get_all(self):
        result = []
        for t in self.repo.yield_sorted():
            self.refresh_task(t)
            result.append(t)
        return result

    def store_result(self, task_id, result):
        task = self.get_task(task_id)
        task.result = result
        task.completed_at = get_timestamp()
        self.repo.update(task_id, task)
        return task

    def get_result(self, task_id):
        task = self.get_task(task_id)
        # if task.status == states.SUCCESS:
        #    res = current_app.AsyncResult(task_id)
        #    return res.result
        if not task.result:
            return task
        return task.result

    def revoke(self, task: Task):
        pass
