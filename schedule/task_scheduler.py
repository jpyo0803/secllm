import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, '..')
if module_path not in sys.path:
    sys.path.append(module_path)

from schedule.topological_sort import TopologicalSort
from schedule.task import Task, TaskSubclassA, TaskSubclassB

class TaskScheduler:
    def __init__(self, graph):
        task_order = TopologicalSort(graph)
        self.tasks = []

        for task_id in task_order:
            next_task_ids = graph[task_id]
            new_task = TaskSubclassA(f'task {task_id}', task_id, next_task_ids) if task_id % 3 == 0 else TaskSubclassB(f'task {task_id}', task_id, next_task_ids)
            self.add_task(new_task)


    def add_task(self, task):
        self.tasks.append(task)

    def run(self):
        for task in self.tasks:
            task()

    def __call__(self):
        self.run()

if __name__ == '__main__':
    # graph = {
    #   0: [],
    #   1: [7, 8],
    #   2: [7],
    #   3: [],
    #   4: [1],
    #   5: [0, 3, 4],
    #   6: [1, 3],
    #   7: [],
    #   8: [],
    # }
    graph = {
        0: [1],
        1: [2, 67],
        2: [3],
        3: [7, 8, 9],
        4: [13],
        5: [14],
        6: [15],
        7: [10],
        8: [11],
        9: [12],
        10: [13],
        11: [14],
        12: [15],
        13: [16],
        14: [17],
        15: [18],
        16: [19],
        17: [20],
        18: [21],
        19: [24],
        20: [25],
        21: [47],
        22: [23],
        23: [24, 25],
        24: [28],
        25: [29],
        26: [38],
        27: [30],
        28: [31],
        29: [32],
        30: [33, 34, 35],
        31: [33, 35],
        32: [34, 35],
        33: [36],
        34: [37],
        35: [41],
        36: [39],
        37: [38],
        38: [39],
        39: [40],
        40: [41],
        41: [42],
        42: [43],
        43: [46],
        44: [56],
        45: [48],
        46: [49],
        47: [50],
        48: [51, 52, 53],
        49: [51, 53],
        50: [52, 53],
        51: [54],
        52: [55],
        53: [59],
        54: [57],
        55: [56],
        56: [57],
        57: [58],
        58: [59],
        59: [60],
        60: [62],
        61: [64],
        62: [63],
        63: [64],
        64: [65],
        65: [66],
        66: [67],
        67: [68],
        68: [69, 90],
        69: [70],
        70: [73, 74],
        71: [77],
        72: [78],
        73: [75],
        74: [76],
        75: [77],
        76: [78],
        77: [79],
        78: [80],
        79: [81],
        80: [82],
        81: [83],
        82: [83],
        83: [85],
        84: [87],
        85: [86],
        86: [87],
        87: [88],
        88: [89],
        89: [90],
        90: [],
    }
    task_scheduler = TaskScheduler(graph)
    task_scheduler()

