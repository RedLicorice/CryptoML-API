from celery import current_app, states
from cryptoml_core.repositories.task_repository import TaskRepository, Task
from typing import List


class TaskService:
    def __init__(self):
        self.repo = TaskRepository()

    def send(self, name, args: dict):
        celery_task = current_app.send_task(name, args=[args])
        task = Task(
            task_id=celery_task.id,
            name=name
        )
        self.repo.create(task)
        return task

    def send_many(self, name, args_list: List[dict]):
        for args in args_list:
            yield self.send(name, args)

    def get_status(self, task_id):
        task = self.repo.find_by_task_id(task_id)
        if task.status not in states.READY_STATES:
            res = current_app.AsyncResult(task_id)
            task.status = res.state
            if res.state != task.status:
                self.repo.update(task.id, task)
        return task

    def get_all(self):
        result = []
        for t in self.repo.yield_sorted():
            if t.status not in states.READY_STATES:
                res = current_app.AsyncResult(t.task_id)
                t.status = res.state
                if res.state != t.status:
                    self.repo.update(t.id, t)
            result.append(t)
        return result

    def get_result(self, task_id):
        task = self.get_status(task_id)
        if task.status == states.SUCCESS:
            res = current_app.AsyncResult(task_id)
            return res.result
        return task

    def revoke(self, task: Task):
        pass
