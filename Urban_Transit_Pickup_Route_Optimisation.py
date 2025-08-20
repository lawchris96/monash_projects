"""
READ ME

Given scenario: a driver must meet a friend who is riding a circular city rail loop, 
arriving exactly when the train reaches a station, while minimising total driving cost 
with earliest arrival used to break ties. This is solved by modelling the network as a 
transit-aware multiverse graph where each node is a (location, minute-in-loop) pair and 
each directed edge encodes a valid road move with its travel time and cost. The friend's 
train position is precomputed for every minute of a full loop. Dijkstra's algorithm runs 
on this multiverse graph using a custom min-heap and a lexicographic priority that considers 
cost first and time second. The solution scans all minutes to find states where the driver 
and train coincide at the same station, selects the best meeting option, reconstructs the 
full driving route from predecessors, and reports the meeting station, ETA, and total cost, 
or clearly indicates when interception is not possible.
 
"""

class Vertex:
    def __init__(self, location: int, time_mod: int):
        """
        Function description: 
        This vertex class represents the vertex in the graph. A vertex that corresponds to a specific location at a 
        specific time step within the cycle of the train loop, used to model movement through both space and time in 
        the multiverse graph.

        Time complexity: O(1)

        Space complexity: O(1)
    
        """
        self.location = location
        self.time_mod = time_mod

    def get_index(self, total_time_taken: int) -> int:
        return self.location * total_time_taken + self.time_mod

    @staticmethod
    def from_index(index: int, total_time_taken: int):
        return Vertex(index // total_time_taken, index % total_time_taken)

class Edge:
    def __init__(self, to_index: int, cost: int, travel_time: int):
        """
        Function description: 
        This edge class represents the edge in the graph. A directed connection between two vertices in 
        the multiverse graph, capturing movement from one location and time to another based on a specific 
        travel cost and duration.

        Time complexity: O(1)

        Space complexity: O(1)
        """
        self.to = to_index
        self.cost = cost
        self.time = travel_time

class MinHeap:
    """
    A minimum heap data structure used as a priority queue in Dijkstra's algorithm.
    This implementation is adapted from the max heap taught in FIT1008 and has been 
    modified to always maintain the smallest element at the root.
    """

    MIN_CAPACITY = 1

    def __init__(self, max_size: int):
        """
        Function Description:
            This class initializes the MinHeap with a specified maximum size.

        Approach Description:
            It creates an internal array of size max_size + 1 to store the elements.
            The array is 1-indexed for easier parent-child calculations.

        Input:
            max_size (int): Maximum number of elements the heap can store.

        Output:
            None

        Time Complexity:
            O(n), where n is max_size

        Space Complexity:
            O(n), where n is max_size
        """
        self.size = 0
        self.data = [None] * (max(self.MIN_CAPACITY, max_size) + 1)

    def __len__(self):
        """
        Function Description:
            This function returns the number of elements currently in the heap.

        Output:
            int: Number of elements in the heap.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.size

    def is_full(self):
        """
        Function Description:
            This function checks whether the heap has reached its maximum capacity.

        Output:
            bool: True if the heap is full, False otherwise.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.size + 1 == len(self.data)

    def rise(self, index: int):
        """
        Function Description:
            This function moves an element upward in the heap until the heap property is restored.

        Input:
            index (int): The index of the element to be moved upward.

        Output:
            None

        Time Complexity:
            O(log n), where n is the number of elements in the heap

        Space Complexity:
            O(1)
        """
        item = self.data[index]
        while index > 1 and item[:3] < self.data[index // 2][:3]:
            self.data[index] = self.data[index // 2]
            index = index // 2
        self.data[index] = item

    def add(self, item: tuple):
        """
        Function Description:
            This function adds a new item into the heap.

        Input:
            item (tuple): The element to insert. Comparison is based on the first three values.

        Output:
            bool: True if the item is added successfully.

        Time Complexity:
            O(log n), where n is the number of elements in the heap

        Space Complexity:
            O(1)
        """
        if self.is_full():
            raise IndexError
        self.size += 1
        self.data[self.size] = item
        self.rise(self.size)
        return True

    def smallest_child(self, index: int):
        """
        Function Description:
            This function returns the index of the smaller child for a given parent node.

        Input:
            index (int): The index of the parent node.

        Output:
            int: The index of the smaller child node.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        if 2 * index == self.size or self.data[2 * index][:3] < self.data[2 * index + 1][:3]:
            return 2 * index
        else:
            return 2 * index + 1

    def sink(self, index: int):
        """
        Function Description:
            This function moves an element downward in the heap until the heap property is restored.

        Input:
            index (int): The index of the element to be moved downward.

        Output:
            None

        Time Complexity:
            O(log n), where n is the number of elements in the heap

        Space Complexity:
            O(1)
        """
        item = self.data[index]
        while 2 * index <= self.size:
            small_child = self.smallest_child(index)
            if self.data[small_child][:3] >= item[:3]:
                break
            self.data[index] = self.data[small_child]
            index = small_child
        self.data[index] = item

    def get_min(self):
        """
        Function Description:
            This function removes and returns the smallest item in the heap.

        Output:
            tuple: The smallest item from the heap.

        Time Complexity:
            O(log n)

        Space Complexity:
            O(1)
        """
        if self.size == 0:
            raise IndexError
        smallest = self.data[1]
        self.size -= 1
        if self.size > 0:
            self.data[1] = self.data[self.size + 1]
            self.sink(1)
        return smallest

    def empty(self):
        """
        Function Description:
            This function checks whether the heap is empty.

        Output:
            bool: True if the heap is empty, False otherwise.

        Time Complexity:
            O(1)

        Space Complexity:
            O(1)
        """
        return self.size == 0

    def push(self, item: tuple):
        """
        Function Description:
            This function pushes a new item into the heap and this is also a wrapper for add.

        Input:
            item (tuple): The item to be inserted.

        Output:
            None

        Time Complexity:
            O(log n)

        Space Complexity:
            O(1)
        """
        self.add(item)

    def pop(self):
        """
        Function Description:
            This function removes and returns the smallest item from the heap and this is also a wrapper for get_min.

        Output:
            tuple: The smallest item in the heap.

        Time Complexity:
            O(log n)

        Space Complexity:
            O(1)
        """
        return self.get_min()


def multiverse_graph(roads, num_places, total_time_taken):
    """
    Function description:
        A multiverse graph is created with this function to simulate the movement of an individual across different 
        places over time. As the train route repeats, each vertex represents a specific location at a specific time 
        within the cycle of the train route. With this structure, shortest-path algorithms can take both travel time 
        and location into account when planning a route.

    Approach description:
        In the cycle, the function creates an adjacency list with a vertex for each possible combination of place and time. 
        The algorithm calculates the time at which the destination would be reached for every road and every possible departure 
        time. A directed edge is then created between the vertices of departure and arrival. The edges of the graph represent 
        valid movements from one point to another taken into account the passage of time. Based on the resulting graph, routing 
        decisions can be made in accordance with the timing of actual movements between locations.

    :Input:
        1. roads:
           A list of tuples, each containing the start location, end location, travel cost, and travel time.
        2. num_places:
           The total number of unique physical locations.
        3. total_time_taken:
           The number of time units in one full cycle of the train route.

    :Output, return or postcondition:
        An adjacency list representing the multiverse graph.
        It contains num_places multiplied by total_time_taken vertices.
        Each vertex has a list of edges showing valid movements between locations over time.

    :Time complexity:
    
    O((|R| + |L|)·T) → O(|R| + |L|)

    Definitions:
    |L| = total number of distinct map locations
    |R| = total number of directed roads in roads
    T   = total_time_taken = sum of all inter-station travel times (≤ 100 minutes, constant)

    :Time complexity analysis:
        1. Allocate adjacency list with |L|·T entries: O(|L|·T)
        2. For each of the |R| roads and each time t in [0…T-1]:
        - Compute departure and arrival indices through Vertex.get_index(): O(1) for each respective operation
        - Constructing one Edge: O(1)
        - Appending to list    :  O(1)
        
        Total: O(|R|·T)
         We know that T is bounded by 100, the factor T is constant and so it can be dropped, giving the final complexity of O(|R| + |L|).

    :Space complexity:

    Input complexity: O(|R| + |L|)
    Auxiliary space complexity: O(|R| + |L|)

    :Space complexity analysis:
        - The input roads list and location count use linear space: O(|R| + |L|).
        - Constructing an adjacency list of |L|·T buckets.
        - Each bucket holds at most one edge per road per time tick.
        - Since T is constant, total auxiliary space will be O(|R| + |L|).

    """
    graph_size = num_places * total_time_taken
    # To initialize the graph as an adjacency list for each vertex found
    graph = []
    for i in range(graph_size):
        graph.append([])

    # In here, it will create time-aware edges, for each road and each possible time
    for from_place, to_place, cost, travel_time in roads: 
        for current_time in range(total_time_taken): 
            # Then it will calculate new time after traveling this road starting at current_time
            next_time = (current_time + travel_time) % total_time_taken

            # To map the (location, time) pairs to vertex indices
            departure_index = Vertex(from_place, current_time).get_index(total_time_taken)
            arrival_index = Vertex(to_place, next_time).get_index(total_time_taken)

            # To add the directed edge to the graph
            graph[departure_index].append(Edge(arrival_index, cost, travel_time))

    return graph

def dijkstra_algorithm(graph, num_places, start_location, total_time_taken):
    """
    Function description:
        In this function, Dijkstra's algorithm is applied to a multiverse graph, where each vertex represents a 
        specific position at a specific time modulo the full train loop duration. In this algorithm, the least-cost 
        path and the time required to reach all other vertices are calculated, starting from the given location at the 
        beginning of the algorithm.

    Approach description:
        This function initializes arrays containing the minimum cost, total time, and previous vertex for every vertex 
        in the graph. Based on cost and time, a custom min-heap is employed to select the next best vertex efficiently. 
        As a result of the algorithm, each reachable vertex is explored iteratively and the shortest known path is updated 
        with a better one when one is discovered. In this way, all states in the graph are mapped from the start point to 
        all other states.

    :Input:
        1. graph:
           The multiverse graph is represented as an adjacency list containing edges for each vertex.
        2. num_places:
           The total number of distinct physical locations/stations.
        3. start_location:
           The integer representing the driver's starting location.
        4. total_time_taken:
           The total time taken that finish one full train loop cycle.

    :Output, return or postcondition:
        A tuple (shortest, time_to_reach, path) where:
            - shortest represents a list containing the least travel cost to each vertex
            - time_to_reach represents a list containing the total time taken to reach each vertex
            - path represents a list of parent indices to reconstruct the driving route

    :Time complexity:
     |L| = number of distinct map locations
     |R| = number of directed roads
     T   = time taken for a full train-loop in minutes (constant ≤ 100)
     V   = |L|·T  : number of time-expanded vertices
     E   = |R|·T  : number of time-expanded edges

    - Vertex initialization           : O(V)
    - First heap insertion            : O(log V)
    Main Dijkstra loop:
     - V removals from heap           : O(V·log V)
     - E edge visits with heap updates: O(E·log V)
    Combined: O(V·log V + E·log V) = O((|L|·T + |R|·T)·log(|L|·T))
    Final: O(|R| log |L|)

    :Time complexity analysis:
        1. Total vertices V = |L|·T and edges E = |R|·T.  
        2. Heap operations dominate: O((V + E)·log V) = O((|L|·T + |R|·T)·log(|L|·T)).  
        3. With T being constant, this reduces to O((|L| + |R|)·log |L|).  
        4. As the specification states that each location must have a outgoing road to another location, thus L = R even in the worse case scenario, and R > L in every other scenario,
           thus, the final bound is O(|R| log |L|).

    :Space complexity:

    Input complexity: O(|R| + |L|)
    Auxiliary space complexity: O(|R| + |L|)

    :Space complexity analysis:
        - The input graph has O(|L|·T) vertices and O(|R|·T) edges.
        - Algorithm uses:
            • shortest, time, and path arrays of size O(|L|·T)
            • min-heap with up to O(|L|·T) entries
        - With T being constant, all auxiliary structures will reduce to O(|R| + |L|).
    """
    graph_size = num_places * total_time_taken 
    # The minimum cost to reach each vertex
    shortest = [float('inf')] * graph_size     
    # The corresponding time taken to each vertex      
    time_to_reach = [float('inf')] * graph_size      
    # For route reconstruction
    path = [None] * graph_size                      

    heap = MinHeap(graph_size + 1)

    start_index = Vertex(start_location, 0).get_index(total_time_taken)
    heap.push((0, 0, start_index))

    shortest[start_index] = 0
    time_to_reach[start_index] = 0

    while not heap.empty():
        cost, time, index = heap.pop()

        # To skip this node if a cheaper path was already found
        if cost > shortest[index]:
            continue

        for edge in graph[index]:
            new_cost = cost + edge.cost
            new_time = time + edge.time

            # To update if cheaper or equally cheap but faster
            if new_cost < shortest[edge.to] or (new_cost == shortest[edge.to] and new_time < time_to_reach[edge.to]):
                shortest[edge.to] = new_cost
                time_to_reach[edge.to] = new_time
                path[edge.to] = index
                heap.push((new_cost, new_time, edge.to))

    return shortest, time_to_reach, path

def intercept(roads, stations, start, friendStart):
    """
    Function description:
        The purpose of this function is to determine the most economical path for a driver to take if he wishes to intercept 
        a friend on a circular train loop. The intercept must take place at an exact time when the friend arrives at the train 
        station. A function returns the interception with the lowest total cost if multiple interceptions share the same cost 
        then the interception with the shortest time wins if multiple interceptions are equal in cost. Through the use of a 
        multiverse graph, space and time are modeled.

    Approach description:
        In the first step, the function determines the total number of locations, and then calculates the full duration of a 
        train loop. After this is accomplished, a multiverse graph is constructed in which each node represents a location at 
        a specific time modulo the loop duration. In this loop, the location of the train is calculated at every time unit. 
        The shortest cost and time between the driver's starting location and every vertex in the multiverse graph are computed 
        using Dijkstra's algorithm. As a result, the function seeks out the first moment when the driver and train are both at 
        the same station at the same time, and reconstructs the path to that point of interception.

    :Input:
        1. roads:
           List of tuples (u, v, cost, time) to represent directed roads between locations.
        2. stations:
           List of tuples (station_location, travel_time) to form the train's circular route.
        3. start:
           The integers is to represent the driver's starting location.
        4. friendStart:
           The integer is to represent the station where the friend starts.

    :Output, return or postcondition:
        A tuple of (cost, time, route) where:
            - cost represents the total travel cost collect by the driver
            - time represents the total time taken to reach the interception point
            - route represents a list of integers representing the driver's path from start to interception
        When interception is not possible, it will return None.

    :Time complexity:

    Definitions:
    |L| = number of distinct map locations
    |R| = number of directed roads
    T   = time taken for a full train-loop in minutes (≤ 100, which is constant)

    - Graph build phase      : O(|R|·T + |L|·T)
    - Dijkstra phase         : O((|R|·T + |L|·T)·log(|L|·T))
    - Station setup, train position simulation, and intercept scan : O(T)
    Combined: O((|R|·T + |L|·T)·log(|L|·T) + |R|·T + |L|·T + T)
    Final: O(|R| log |L|)

    :Time complexity analysis:
        1. T is constant, so |R|·T + |L|·T → O(|R| + |L|), and log(|L|·T) → O(log |L|).
        2. Station/setup and scan overhead operations is O(T) so it becomes O(1). The linear term O(|R| + |L|) also grows more slowly than O(|R|·log|L|).
        3. Dijkstra's part O(|R|·log|L|) grows faster than the linear graph-build term as |R| and |L| increase, making it dominates.
        4. So, the overall complexity is O(|R|·log|L|).

    :Space complexity:

    Input complexity: O(|R| + |L|)
    Auxiliary space complexity: O(|R| + |L|)

    :Space complexity analysis:
        - Input includes roads (|R|) and stations (≤ |L|).
        - Builds:
            • station_index, sequence, and time arrays → O(|L|)
            • train_at_time table → O(T)
            • multiverse graph → O(|L|·T + |R|·T)
            • Dijkstra arrays and heap → O(|L|·T)
        - Since T is constant, total auxiliary space remains O(|R| + |L|).
    """
    # To calculate the highest numbered location for array bounds
    max_location = 0
    for u, v, _, _ in roads:
        max_location = max(max_location, u + 1, v + 1)
    for station in stations:
        max_location = max(max_location, station[0] + 1)

    # Initializing arrays to store train route information
    # Maps station location to index in loop
    station_index = [-1] * max_location      
    # The sequence of stations in the loop    
    station_sequence = []                      
    # The travel time to next station in the loop
    time_to_next_station = []        
    # The total time of a complete loop            
    total_time_taken = 0                         

    # To populate train loop route and calculate total loop duration
    i = 0
    while i < len(stations):
        loc = stations[i][0]
        t = stations[i][1]
        station_index[loc] = i
        station_sequence.append(loc)
        time_to_next_station.append(t)
        total_time_taken += t
        i += 1

    # To build the multiverse graph with nodes represented as (location, time_mod)
    graph = multiverse_graph(roads, max_location, total_time_taken)

    # To simulate the train's location at each time in its full loop
    train_at_time = [-1] * total_time_taken
    index = station_index[friendStart]
    t = 0
    while t < total_time_taken:
        train_at_time[t] = station_sequence[index]
        t += time_to_next_station[index]
        index = (index + 1) % len(station_sequence)

    # To run Dijkstra’s algorithm on the multiverse graph starting from the driver's location
    shortest, time_to_reach, path = dijkstra_algorithm(
        graph, max_location, start, total_time_taken)

    # To identify the best possible interception point based on cost and time
    best = None
    for loop_minute in range(total_time_taken):
        station = train_at_time[loop_minute]
        if station == -1:
            continue

        # To convert (station, time) to graph index
        idx = Vertex(station, loop_minute).get_index(total_time_taken)
        cost = shortest[idx]
        time = time_to_reach[idx]

        # Ignore unreachable nodes
        if cost == float('inf'):
            continue

        # To select if it's the first option, cheaper, or equally cheap then faster
        if best is None or cost < best[0] or (cost == best[0] and time < best[1]):
            best = (cost, time, idx)

    # Return None if no valid interception found
    if best is None:
        return None

    # To reconstruct the path from start to intercept node using predecessor array
    cost, time, idx = best
    route = []
    while idx is not None:
        v = Vertex.from_index(idx, total_time_taken)
        route.insert(0, v.location)
        idx = path[idx]

    return (cost, time, route)
