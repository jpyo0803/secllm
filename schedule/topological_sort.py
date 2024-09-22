
def Dfs(graph, u, visited, stack):
  visited[u] = True

  for v in graph[u]:
    if visited[v] == False:
      Dfs(graph, v, visited, stack)

  stack.append(u)

def TopologicalSortBasedOnDfs(graph):
  n = len(graph)

  visited = [False] * n
  stack = []

  for u in range(n):
    if visited[u] == False:
      Dfs(graph, u, visited, stack)

  stack.reverse()
  return stack

def TopologicalSort(graph):
  return TopologicalSortBasedOnDfs(graph)


if __name__ == '__main__':
    graph = {
      0: [],
      1: [7, 8],
      2: [7],
      3: [],
      4: [1],
      5: [0, 3, 4],
      6: [1, 3],
      7: [],
      8: [],
    }
    # graph = {
    #   0: [1, 3],
    #   1: [2],
    #   2: [3],
    #   3: [4],
    #   4: [5, 7],
    #   5: [6],
    #   6: [7],
    #   7: [],
    # }
    print(TopologicalSort(graph))





