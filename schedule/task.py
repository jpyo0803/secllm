
from typing import Any


class Task:
    def __init__(self, name: str, task_id : int, next_task_ids: list[int]):
        self.name = name
        self.task_id = task_id
        self.next_task_ids = next_task_ids

    def run(self):
        print(f"Task: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

class TaskSubclassA(Task):
    def __init__(self, name: str, task_id : int, next_task_ids: list[int]):
        super().__init__(name, task_id, next_task_ids)

    def run(self):
        print(f"TaskSubclassA: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

class TaskSubclassB(Task):
    def __init__(self, name: str, task_id : int, next_task_ids: list[int]):
        super().__init__(name, task_id, next_task_ids)

    def run(self):
        print(f"TaskSubclassB: {self.name, self.task_id, self.next_task_ids}")

    def __call__(self):
        self.run()

if __name__ == '__main__':
    task = Task("task 0", 0, [1, 2])
    task()