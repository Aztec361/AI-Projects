import heapq




def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    n = int(len(from_state) ** 0.5)
    from_state = [from_state[i:i+n] for i in range(0, len(from_state), n)]
    to_state = [to_state[i:i+n] for i in range(0, len(to_state), n)]
    distance = 0
    for i in range(n):
        for j in range(n):
            if from_state[i][j] != 0:
                goal_i, goal_j = next((k, l) for k in range(n) for l in range(n) if to_state[k][l] == from_state[i][j])
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance




def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 

    """
    empty_tile1 = state.index(0)
    empty_tile2 = state.index(0, empty_tile1 + 1)
    succ_states = []

    if empty_tile1 % 3 != 0:
        succ_state = state.copy()
        succ_state[empty_tile1], succ_state[empty_tile1 - 1] = succ_state[empty_tile1 - 1], succ_state[empty_tile1]
        succ_states.append(succ_state)

    if empty_tile1 % 3 != 2:
        succ_state = state.copy()
        succ_state[empty_tile1], succ_state[empty_tile1 + 1] = succ_state[empty_tile1 + 1], succ_state[empty_tile1]
        succ_states.append(succ_state)

    if empty_tile1 // 3 != 0:
        succ_state = state.copy()
        succ_state[empty_tile1], succ_state[empty_tile1 - 3] = succ_state[empty_tile1 - 3], succ_state[empty_tile1]
        succ_states.append(succ_state)

    if empty_tile1 // 3 != 2:
        succ_state = state.copy()
        succ_state[empty_tile1], succ_state[empty_tile1 + 3] = succ_state[empty_tile1 + 3], succ_state[empty_tile1]
        succ_states.append(succ_state)

    if empty_tile2 % 3 != 0:
        succ_state = state.copy()
        succ_state[empty_tile2], succ_state[empty_tile2 - 1] = succ_state[empty_tile2 - 1], succ_state[empty_tile2]
        succ_states.append(succ_state)

    if empty_tile2 % 3 != 2:
        succ_state = state.copy()
        succ_state[empty_tile2], succ_state[empty_tile2 + 1] = succ_state[empty_tile2 + 1], succ_state[empty_tile2]
        succ_states.append(succ_state)

    if empty_tile2 // 3 != 0:
        succ_state = state.copy()
        succ_state[empty_tile2], succ_state[empty_tile2 - 3] = succ_state[empty_tile2 - 3], succ_state[empty_tile2]
        succ_states.append(succ_state)

    if empty_tile2 // 3 != 2:
        succ_state = state.copy()
        succ_state[empty_tile2], succ_state[empty_tile2 + 3] = succ_state[empty_tile2 + 3], succ_state[empty_tile2]
        succ_states.append(succ_state)

    for i in succ_states:
        if i == state:
            succ_states.remove(i)

    return sorted(succ_states)

def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    pq = []
    visited = []
    visited_states = []
    max_length = 1
    g = 0
    h = get_manhattan_distance(state)
    cost = g + h
    parent_index = -1

    heapq.heappush(pq, (cost, state, (g, h, parent_index)))
    visited_element = heapq.heappop(pq)
    visited.append(visited_element)
    visited_states.append(visited_element[1])

    while visited_element[1] != goal_state:
        for each in get_succ(visited_element[1]):
            if each not in visited_states:
                g = visited_element[2][0] + 1
                h = get_manhattan_distance(each)
                cost = g + h
                parent_index = visited.index(visited_element)
                heapq.heappush(pq, (cost, each, (g, h, parent_index)))
        if len(pq) > max_length:
            max_length = len(pq)
        visited_element = heapq.heappop(pq)
        visited.append(visited_element)
        visited_states.append(visited_element[1])

    # printing path
    path = []
    while visited_element[2][2] != -1:
        path.append(visited_element[1])
        visited_element = visited[visited_element[2][2]]

    path.append(visited_element[1])
    visited_element = visited[visited_element[2][2]]

    path.reverse()

    for each in path:
        print(each, "h={}".format(get_manhattan_distance(each)), "moves: {}".format(path.index(each)))
    print("Max queue length: {}".format(max_length))


def print_state(state, h, num_moves):

     print(state, f'h={h} moves: {num_moves}')

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([4,3,0,5,1,6,7,2,0])
    print()
