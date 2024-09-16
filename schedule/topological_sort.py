
def Dfs(graph, u, visited, stack):
  visited[u] = True

  for v in graph[u]:
    if visited[v] == False:
      Dfs(graph, v, visited, stack)

  stack.append(u)

def TopologicalSortBasedOnDfs(graph):
  n = 0
  for u, v in graph:
    n = max(n, u, v)
  n += 1

  processed_graph = [[] for _ in range(n)]
  for u, v in graph:
    processed_graph[u].append(v)

  visited = [False] * n
  stack = []

  for u in range(n):
    if visited[u] == False:
      Dfs(processed_graph, u, visited, stack)

  stack.reverse()
  return stack

def TopologicalSort(graph):
  return TopologicalSortBasedOnDfs(graph)


if __name__ == '__main__':
    graph = [
      (5, 0),
      (5, 3),
      (2, 7),
      (5, 4),
      (4, 1),
      (1, 7),
      (6, 3),
      (6, 1),
      (1, 8),
    ]
    print(TopologicalSort(graph))





