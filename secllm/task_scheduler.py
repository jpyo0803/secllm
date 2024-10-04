import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
module_path = os.path.join(current_dir, '..')
if module_path not in sys.path:
    sys.path.append(module_path)

from secllm.topological_sort import TopologicalSort

import secllm.task

class TaskScheduler:
    def __init__(self, graph, secllm_cpp_wrapper, model_info, thread_pool, time_collector):
        task_order = TopologicalSort(graph)
        self.tasks_per_layer = []
        self.thread_pool = thread_pool 

        for layer_idx in range(model_info.config.num_hidden_layers):
            tasks = []
            for task_id in task_order:
                next_task_ids = graph[task_id]
                class_name = f'Task{task_id}'

                task_class = getattr(secllm.task, class_name, None)
                if task_class:
                    new_task = task_class(f'task {task_id}', layer_idx, task_id, next_task_ids, secllm_cpp_wrapper, model_info, time_collector)
                    tasks.append(new_task)
                else:
                    assert False, f'Class {class_name} not found'
            self.tasks_per_layer.append(tasks)

    def run(self, layer_idx):
        # for task in self.tasks_per_layer[layer_idx]:
            # task()

        # copy the tasks into data structures that can efficiently run a ready task (is_ready()) and remove it from the list of tasks
        copied_tasks = self.tasks_per_layer[layer_idx].copy()

        # run the tasks if ready and remove them from the list

        while copied_tasks:
            for task in copied_tasks:
                if task.is_ready():
                    # print(f'Enqueue the task: {task.task_id}')
                    # task()
                    # task.print_info()
                    self.thread_pool.enqueue_task(task)
                    copied_tasks.remove(task)
                else:
                    pass
                    # print(f'Task {task.task_id} not ready')

    def __call__(self, layer_idx):
        self.run(layer_idx)

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
    # write the graph to a file
    with open('dependency_graph.txt', 'w') as f:
        for key, value in graph.items():
            f.write(f'{key}: {value}\n')

    from secllm_cpp.secllm_cpp_wrapper import SecLLMCppWrapper

    secllm_cpp_wrapper = SecLLMCppWrapper(32)

    task_scheduler = TaskScheduler(graph, secllm_cpp_wrapper)
    task_scheduler()

'''
Reference Order

Task84: ('task 84', 84, [87])
Task72: ('task 72', 72, [78])
Task71: ('task 71', 71, [77])
Task61: ('task 61', 61, [64])
Task45: ('task 45', 45, [48])
Task48: ('task 48', 48, [51, 52, 53])
Task44: ('task 44', 44, [56])
Task27: ('task 27', 27, [30])
Task30: ('task 30', 30, [33, 34, 35])
Task26: ('task 26', 26, [38])
Task22: ('task 22', 22, [23])
Task23: ('task 23', 23, [24, 25])
Task6: ('task 6', 6, [15])
Task5: ('task 5', 5, [14])
Task4: ('task 4', 4, [13])
Task0: ('task 0', 0, [1])
Task1: ('task 1', 1, [2, 67])
Task2: ('task 2', 2, [3])
Task3: ('task 3', 3, [7, 8, 9])
Task9: ('task 9', 9, [12])
Task12: ('task 12', 12, [15])
Task15: ('task 15', 15, [18])
Task18: ('task 18', 18, [21])
Task21: ('task 21', 21, [47])
Task47: ('task 47', 47, [50])
Task50: ('task 50', 50, [52, 53])
Task52: ('task 52', 52, [55])
Task55: ('task 55', 55, [56])
Task56: ('task 56', 56, [57])
Task8: ('task 8', 8, [11])
Task11: ('task 11', 11, [14])
Task14: ('task 14', 14, [17])
Task17: ('task 17', 17, [20])
Task20: ('task 20', 20, [25])
Task25: ('task 25', 25, [29])
Task29: ('task 29', 29, [32])
Task32: ('task 32', 32, [34, 35])
Task34: ('task 34', 34, [37])
Task37: ('task 37', 37, [38])
Task38: ('task 38', 38, [39])
Task7: ('task 7', 7, [10])
Task10: ('task 10', 10, [13])
Task13: ('task 13', 13, [16])
Task16: ('task 16', 16, [19])
Task19: ('task 19', 19, [24])
Task24: ('task 24', 24, [28])
Task28: ('task 28', 28, [31])
Task31: ('task 31', 31, [33, 35])
Task35: ('task 35', 35, [41])
Task33: ('task 33', 33, [36])
Task36: ('task 36', 36, [39])
Task39: ('task 39', 39, [40])
Task40: ('task 40', 40, [41])
Task41: ('task 41', 41, [42])
Task42: ('task 42', 42, [43])
Task43: ('task 43', 43, [46])
Task46: ('task 46', 46, [49])
Task49: ('task 49', 49, [51, 53])
Task53: ('task 53', 53, [59])
Task51: ('task 51', 51, [54])
Task54: ('task 54', 54, [57])
Task57: ('task 57', 57, [58])
Task58: ('task 58', 58, [59])
Task59: ('task 59', 59, [60])
Task60: ('task 60', 60, [62])
Task62: ('task 62', 62, [63])
Task63: ('task 63', 63, [64])
Task64: ('task 64', 64, [65])
Task65: ('task 65', 65, [66])
Task66: ('task 66', 66, [67])
Task67: ('task 67', 67, [68])
Task68: ('task 68', 68, [69, 90])
Task69: ('task 69', 69, [70])
Task70: ('task 70', 70, [73, 74])
Task74: ('task 74', 74, [76])
Task76: ('task 76', 76, [78])
Task78: ('task 78', 78, [80])
Task80: ('task 80', 80, [82])
Task82: ('task 82', 82, [83])
Task73: ('task 73', 73, [75])
Task75: ('task 75', 75, [77])
Task77: ('task 77', 77, [79])
Task79: ('task 79', 79, [81])
Task81: ('task 81', 81, [83])
Task83: ('task 83', 83, [85])
Task85: ('task 85', 85, [86])
Task86: ('task 86', 86, [87])
Task87: ('task 87', 87, [88])
Task88: ('task 88', 88, [89])
Task89: ('task 89', 89, [90])
Task90: ('task 90', 90, [])

'''