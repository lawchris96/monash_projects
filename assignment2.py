from collections import deque

class Edge:
    """
    Function Description:
    
    This class create edges to represent a directed connection from vertex source to target with capacity bounds

    Approach Description:

    It stores the four input values as attributes, then initialize current_flow to 0 then leave reverse set to None 
    and will be assigned when the backward edge is formed
        
    Input:
        source       = int, the index of the start vertex
        target       = int, the index of the end vertex
        lower_bound  = int, the minimum flow that go through the edge
        upper_bound  = int, the maximum flow allowed through the edge

    Output:
        None

    Time and Space Complexity:
        - O(1) for both, since this merely just to assign a fixed number of fields
    """
    def __init__(self, source, target, lower_bound, upper_bound):
        # Assigning to fields
        self.source, self.target, self.lower_bound, self.upper_bound = (
            source, target, lower_bound, upper_bound
        )
        # Initializing the flow and reverse pointer exactly as before
        self.current_flow = 0
        self.reverse = None

class Vertex:
    NODE_COUNTER = 0

    def __init__(self, data, idx=None):
        """
        Function Description:

        This class is to initialise a vertex by assigning it a unique identifier while storing the provided data

        Approach Description:

        1. When idx is provided, assign it to self.idx, else assign the next available index from NODE_COUNTER
        2. When idx was not provided, increment then occur in NODE_COUNTER
        3. Storing the data then initialize self.edges as an empty list to hold outgoing Edge objects

        Input:
            data = the data for this vertex like class ID, timeslot ID, or None
            idx  = int or None, if provided use as this vertex index, else use NODE_COUNTER

        Output:
            None

        Time and Space Complexity:
            - O(1) for both, since this only assigns a fixed number of attributes
        """
        # Using provided idx or take the next available counter
        self.idx = idx if idx is not None else Vertex.NODE_COUNTER
        if idx is None:
            # Incrementing the global counter only when idx was not given
            Vertex.NODE_COUNTER += 1

        self.data = data
        # Initializing adjacency list for outgoing edges
        self.edges = []

    def add_edge(self, edge):
        """
        Function Description:

        This function appends an edge to vertex adjacency list

        Approach Description:

        It takes the given edge object and append it to self.edges

        Input:
            edge = Edge, the given directed edge to add as an outgoing connection

        Output:
            None

        Time and Space Complexity:
            - O(1) for both, as appending to a list takes constant time and space
        """
        self.edges.append(edge)


def build_graph_helper(n, m, timePreferences, proposedClasses):
    """
    Function Description:

    This function constructs the flow network graph to assign each of n students to classes that have 
    fixed timeslots and capacity bounds. It then creates and connects source, student, timeslot, class, 
    and sink vertices, and records all edges that enforce each class minimum and maximum capacitie

    Approach Description:

    1. Resetting the vertex counter so that every time the code runs, the vertices start numbering from zero
    2. Creating a source vertex with index zero and putting it into the list of all vertices
    3. For each student, it will then perform these steps:
        - Making a new vertex for that student and add it to the list of all vertices
        - Linking the source to that student by creating an edge that allows exactly one unit of flow and a 
          reversing edge that does not allow the flow
        - Adding the forward edge to the source list
    4. Setting up an array of length 20 to hold timeslot vertices as it is all empty at the beginning
    5. Creating a sink vertex at the upcoming available index while storing it in the list of all vertices
    6. For every class, it will then follow these steps:
        - Getting its timeslot identifier and the minimum and maximum number of students it can take
        - If no vertex exists yet for that timeslot, it then creates one and add it to both the list and the array
        - Making an edge from the timeslot vertex to the class vertex that must carry at least the minimum number 
          of students and at most the maximum, and adding its reverse edge with zero capacity
        - Adding the forward edge to the timeslot edges and mark it as the critical edge
        - Creating another edge from the class vertex to the sink with the same minimum and maximum limits and its 
          zero capacity reverse
        - Putting that forward edge in the class edge list and mark it as critical
    7. For each student again, it will then do these actions next:
        - Looking up their top five timeslot choices in timePreferences
        - For each choice that has an existing timeslot vertex, create a one unit edge from the student to that 
          timeslot plus a zero capacity reverse
        - Adding just the forward edge to the student list
    8. Reordering the edges leaving the source using a simple bubble sort so that students with less outgoing choices appear first
    9. Returning the source vertex, the list of all vertices, the list of student vertices, the array of timeslot vertices, 
       the sink vertex, and the list of critical edges

    Input:
        n                   = the number of students
        m                   = the number of classes
        timePreferences     = the list of n sublists, each containing 5 preferred timeslot IDs for the corresponding student
        proposedClasses     = the list of m triples, defining each class assigned timeslot and capacity bounds

    Output:
        source              = the source vertex from which all student flow starts
        all_vertices        = the list of every vertex object created like source, students, timeslots, classes, and sink
        students            = the list of the n student vertex objects
        timeslot_nodes      = the length of 20 list of vertex objects corresponding to timeslots where it will be None if unused
        sink                = the sink vertex where class flow ends
        critical_edges      = the list of all forward edges that enforce each class bounds

    Time Complexity:
        Best Case:  O(n^2)
        Analysis:
            - Building the graph with all vertices and edges takes O(n + m)
            - Bubble sorting the source to student edges always costs O(n^2) time, even if every student has similar choices
            - The brief states that M is a positive integer. In the extreme case where every class holds exactly one student,
              M equals N. In every other situation M is smaller than N, so any complexity terms containing N dominate those 
              that contain M.
            - No other part of the helper costs more than O(n + m) so total time is O(n^2 + m) and since m = O(n), this became O(n^2)

        Worst Case:  O(n^2)
        Analysis:
            - Graph construction still costs O(n + m) and bubble sort still costs O(n^2)
            - Nothing skips or short cuts these steps, so total time remains O(n^2 + m), which collapses to O(n^2) when m = O(n)

    Auxiliary Space Complexity:
        Best Case:  O(n)
        Analysis:
            - Storing all vertices requires O(n + m) space while all forward and reverse edges also uses O(n + m) space
            - Helper lists each hold O(n + m) items and bubble sort works in place and needs only O(1) extra space
            - Since m = O(n) based on the specification, total space will be O(n)

        Worst Case:  O(n)
        Analysis:
            - Even for the largest input, vertices, edges, and helper lists together take O(n + m) space, which becomes O(n) when m = O(n)
            - No other structure grows beyond this, and in place sorting adds only constant space
        
        Input Complexity:
        Size:  O(n + m)
        Analysis:
            - timePreferences holds n inner lists, each with exactly 20 integers which is a fixed constant 
              so reading this structure is O(n)
            - proposedClasses holds m entries, each with three integers so reading this structure is O(m)
            - No other input argument grows with problem size so adding these two parts gives a total input size of O(n + m)

    """
    # Resetting the vertex index counter so that new vertices start from 0
    Vertex.NODE_COUNTER = 0

    # Creating a source vertex with index 0 to represent the starting point of the flow network
    source = Vertex(data=None, idx=0)
    all_vertices = [source]

    # In here, for each student, it creates a vertex and add an edge from source to that student with capacity 1
    students = []
    for sid in range(n):
        # Creating a vertex to represent student 'sid'
        student_v = Vertex(data=sid, idx=len(all_vertices))
        students.append(student_v)
        all_vertices.append(student_v)

        # Adding an edge from the source to this student, allowing at most one unit of flow
        edge_src_to_student = Edge(source=source.idx, target=student_v.idx, lower_bound=0, upper_bound=1)
        edge_student_to_src = Edge(source=student_v.idx, target=source.idx, lower_bound=0, upper_bound=0)
        edge_src_to_student.reverse = edge_student_to_src
        edge_student_to_src.reverse = edge_src_to_student

        source.add_edge(edge_src_to_student)

    # Initializing a list for up to 20 timeslot vertices while None indicates that a slot is not yet used
    timeslot_nodes = [None] * 20

    # Creating a sink vertex at the next available index to serve as the end point of the flow network
    sink = Vertex(data=None, idx=len(all_vertices))
    all_vertices.append(sink)

    # Then, for each class, it sets up its vertex and connect the appropriate timeslot and sink with capacity constraints
    critical_edges = []
    for class_id in range(m):
        slot_id, min_cap, max_cap = proposedClasses[class_id]

        # Creating a vertex to represent the class
        class_v = Vertex(data=class_id, idx=len(all_vertices))
        all_vertices.append(class_v)

        # When no vertex exists yet for the class timeslot, it will create one then register it
        if timeslot_nodes[slot_id] is None:
            timeslot_v = Vertex(data=slot_id, idx=len(all_vertices))
            timeslot_nodes[slot_id] = timeslot_v
            all_vertices.append(timeslot_v)
        timeslot_v = timeslot_nodes[slot_id]

        # Connecting the timeslot vertex to the class vertex, imposing the class minimum and maximum student capacity
        edge_timeslot_to_class = Edge(source=timeslot_v.idx, target=class_v.idx, lower_bound=min_cap, upper_bound=max_cap)
        edge_class_to_timeslot = Edge(source=class_v.idx, target=timeslot_v.idx, lower_bound=0, upper_bound=0)
        edge_timeslot_to_class.reverse = edge_class_to_timeslot
        edge_class_to_timeslot.reverse = edge_timeslot_to_class

        timeslot_v.add_edge(edge_timeslot_to_class)
        critical_edges.append(edge_timeslot_to_class)

        # Adding an edge from the class vertex to the sink with the same capacity constraints 
        # just to ensure the total number of students in the class stays within the bounds
        edge_class_to_sink = Edge(source=class_v.idx, target=sink.idx, lower_bound=min_cap, upper_bound=max_cap)
        edge_sink_to_class = Edge(source=sink.idx, target=class_v.idx, lower_bound=0, upper_bound=0)
        edge_class_to_sink.reverse = edge_sink_to_class
        edge_sink_to_class.reverse = edge_class_to_sink

        class_v.add_edge(edge_class_to_sink)
        critical_edges.append(edge_class_to_sink)

    # For each student, it adds edges to their top 5 preferred timeslot vertices
    for sid in range(n):
        student_v = students[sid]
        for rank in range(5):
            slot_id = timePreferences[sid][rank]
            timeslot_v = timeslot_nodes[slot_id]
            if timeslot_v:
                # Allowing a student to send one unit of flow to this preferred timeslot
                edge_student_to_timeslot = Edge(source=student_v.idx, target=timeslot_v.idx, lower_bound=0, upper_bound=1)
                edge_timeslot_to_student = Edge(source=timeslot_v.idx, target=student_v.idx, lower_bound=0, upper_bound=0)
                edge_student_to_timeslot.reverse = edge_timeslot_to_student
                edge_timeslot_to_student.reverse = edge_student_to_timeslot

                student_v.add_edge(edge_student_to_timeslot)

    # Applying bubble sort to reorder edges from the source so that students with fewer available timeslot options are first processed 
    src_edges = source.edges
    L = len(src_edges)
    for i in range(L):
        for j in range(L - i - 1):
            a = src_edges[j].target
            b = src_edges[j + 1].target
            # Comparing how many outgoing edges each student vertex has
            if len(all_vertices[a].edges) > len(all_vertices[b].edges):
                src_edges[j], src_edges[j + 1] = src_edges[j + 1], src_edges[j]

    return source, all_vertices, students, timeslot_nodes, sink, critical_edges

class FlowNetwork:
    """
    The FlowNetwork class represents a flow network where data flows from a source to a sink through 
    a set of vertices and directed edges with capacity constraints
     
    It supports computing the maximum possible flow using the Ford Fulkerson algorithm, 
    which repeatedly uses Breadth First Search (BFS) to find augmenting paths in the residual graph.

    The class manages graph structure, updates flows, and tracks capacity via forward and reverse edges.
    """
    def __init__(self, vertices):
        """
        Function Description:  
        
        This function initializes a FlowNetwork with a given list of vertices and constructs 
        an adjacency list to represent connections between them

        Approach Description:  
        1. Storing the list of vertex objects in self.vertices  
        2. Determining the total number of vertices
        3. Creating an empty list of length size, where each element is itself an empty list, to serve as the adjacency list
        4. Iterating through each vertex and its edges
        5. For each edge, appending the target index to the source list and the source index to the target list to indicate a connection

        Input:  
        vertices = the list of vertex objects where each Vertex contains a list of edges

        Output:  
        None

        Time Complexity:  

        V = the number of vertices in the graph
        E = the number of directed edges in the graph

        Best Case:  O(V + E)
        Analysis:
            - Constructing the outer list requires O(V) time only 
            - At each of the E edges is visited exactly once, and two append operations are performed per edge, giving O(E)
            - No other operations is dominating the steps

        Worst Case: O(V + E)
        Analysis:
            - Even when the graph is dense (E = O(V^2)), the constructor still processes each edge only once, 
              so the total work keeps at O(V + E)

        Auxiliary Space Complexity:  
        Best Case:  O(V + E)
        Analysis:
            - The adjacency list stores one list per vertex and one entry per directed edge, giving O(V + E)
            - Then, temporary loop variables use only constant space

        Worst Case: O(V + E)
        Analysis:
            - In the densest graph the adjacency list holds O(V^2) edge endpoints, which is still O(V + E)

        Input Complexity:  
        Size: O(V + E)
        Analysis:
            - The constructor gets the vertices list, containing V vertex objects and their embedded E edge objects
            - No additional input is read, so the total size of the data the constructor is O(V + E)
        """
        self.vertices = vertices
        size = len(vertices)

        # Creating empty neighbour lists
        self.adj_list = [[] for i in range(size)]

        # Filling the adjacency lists from the per vertex edge collections
        for v in vertices:
            for e in v.edges:
                self.adj_list[e.source].append(e.target)
                self.adj_list[e.target].append(e.source)

    def bfs(self, source_idx, sink_idx, parent_edge):
        """
        To build the bfs function, I have done research online and adapted the methods from the website below.
        website: https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/

        Function Description:
        
        This function finds if there is a path from the source to the sink using Breadth First Search on the residual graph.
        When such a path exists, it will fill the parent_edge list to reconstruct the path.

        Approach Description:

        1. Initializing a visited list and a queue with the source index. Mark the source as visited
        2. While the queue is not empty:
            - Removing the current vertex from the queue
            - For each outgoing edge from the current vertex:
                - If the edge leads to an unvisited vertex and has available capacity:
                    - Mark the vertex as visited
                    - Store the edge in parent_edge to trace the path
                    - If the sink is reached, return True instantly, else, add the next vertex to the queue
        3. If the queue empties without reaching the sink, return False

        Input:
            self.vertices   = the list of Vertex objects, each containing edges with capacity and flow values
            source_idx      = the index of the source node in self.vertices
            sink_idx        = the index of the sink node in self.vertices
            parent_edge     = the list used to store the edge leading to each vertex from the BFS path

        Output:
            Boolean = True when a path exists from source to sink, else False.

        Time Complexity:

            V = the number of vertices in the graph
            E = the number of directed edges in the graph

            Best Case:  O(V + E)
            Analysis:
                - The BFS visits each vertex at most once and scans every outgoing edge also at most once
                - When the sink is discovered early, some vertices or edges may be skipped, but the bound remains O(V + E)

            Worst Case: O(V + E)
            Analysis:
                - The search can visit every vertex and scan every edge exactly once, so total work is proportional to O(V + E)

        Auxiliary Space Complexity:
            Best Case: O(V)
            Analysis:
                - The visited list is size V and the queue may hold only a few vertices when the sink is found early, 
                  but its maximum possible size is still bounded by V
                - parent_edge is then supplied by the caller and already counts as V, so no extra beyond that list

            Worst Case: O(V)
            Analysis:
                - In the worst case the queue can fill with every vertex, so it also uses space V since 
                  no other large structures are created, the peak extra memory will be O(V)

        Input Complexity:
            Size: O(V)
            Analysis:
                - The routine receives two integers and it also receives parent_edge, an array already sized V
                - It does not read any other data, so the input size is O(V)
        """
        visited = [False] * len(self.vertices)
        queue   = deque([source_idx])
        visited[source_idx] = True
        while queue:
            u = queue.popleft()
            # Exploring every outgoing edge of u in the residual graph
            for e in self.vertices[u].edges:
                if not visited[e.target] and (e.upper_bound - e.current_flow) > 0:
                    visited[e.target] = True
                    parent_edge[e.target] = e

                    # If the sink is found, we have an augmenting path
                    if e.target == sink_idx:
                        return True
                    queue.append(e.target)

        return False
    """
    Reference for bfs function:

    GeeksforGeeks. (2023, June 1). FordFulkerson Algorithm for maximum Flow Problem. GeeksforGeeks. 
        https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
    """

    def ford_fulkerson(self, source_idx, sink_idx):
        """
        To build the ford fulkerson function, I have done research online and adapted the methods from the website below.
        website: https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/

        Function Description:
        
        This function computes the maximum flow from source to sink in a flow network using the Ford Fulkerson method, 
        When BFS is used in Ford Fulkerson it will be known as Edmonds-Karp algorithm where BFS will always picks a path 
        with minimum number of edges (GeeksforGeeks, 2023).

        Approach Description:

        1. Setting total_flow to 0, preparing a parent_edge array to record the BFS found path
        2. Using BFS on the residual graph to populate parent_edge for each reachable vertex until the sink is reached
        3. Walking backward from sink to source via parent_edge and recording the minimum residual capacity among those edges
        4. Pushing the bottleneck amount through each edge on the path, updating both the forward and reverse edge flows
        5. Adding the bottleneck value to total_flow and reinitialise the parent_edge for the next BFS iteration
        6. Repeat steps 2 to 5 until BFS fails to reach the sink then return total_flow as the maximum flow

        Input:
            self.vertices   = the list of Vertex objects and each Vertex has an .edges list of Edge objects
                              while each Edge has source, target, upper_bound, current_flow, and reverse reference
            source_idx      = the index of source vertex in self.vertices
            sink_idx        = the index of sink vertex in self.vertices

        Output:
            Integer total_flow = the maximum flow from source_idx to sink_idx

        Time Complexity:

        V = the number of vertices in the graph  
        E = the number of directed edges in the graph  
        F = the value of the returned maximum flow 

        Time Complexity:
            Worst Case: O(V + E)
            Analysis:
                - The classic Edmonds Karp time complexity of O(V E^2) assumes each augmentation uses a BFS that runs in O(V^2) time
                - This is based on representing the residual graph as an adjacency matrix, where scanning a row of neighbours 
                  takes O(V) time
                - In my implementation, the residual graph is stored using adjacency lists instead so each vertex enters 
                  the BFS queue once, contributing O(V) operations
                - Each edge will be examine once during the BFS, contributing O(E) operations as no other operation dominates 
                  these, the total cost of one BFS now becomes O(V + E)
                - The overall Edmonds Karp bound remains O(V E^2), as the number of augmenting paths is still O(V E)
                - Only the per search cost improves from O(V^2) to O(V + E) due to the change in graph representation

        Auxiliary Space Complexity:
            Worst Case:
            Analysis:
                - This implementation uses BFS to locate augmenting paths in the residual graph. Each BFS requires temporary 
                  structures that grow with the number of vertices:
                    - The queue used during traversal holds at most one entry per vertex, giving O(V) space
                    - The parent_edge array stores the edge leading into each vertex along the augmenting path, 
                      also requiring O(V) space
                    - The visited tracking mechanism, if used, consumes an additional O(V) space
                - Now, no additional structures grow with the number of edges or the flow values and all edge and 
                  vertex information is pre-allocated within the graph.  
                - Lastly, the total auxiliary space used in the worst case is proportional to the number of vertices, 
                  giving a space complexity of O(V).

        Input Complexity:  
        Size: O(V + E)  
        Analysis:  
            - The method receives two integers and accesses the pre built graph stored in self.vertices, 
              which contains V vertices and E edges and no additional input is read
        """
        total_flow  = 0
        parent_edge = [None] * len(self.vertices)
        while self.bfs(source_idx, sink_idx, parent_edge):
            bottleneck = float("inf")
            v = sink_idx
            # Walking backward from sink to source to find minimum residual capacity
            while v != source_idx:
                e = parent_edge[v]
                rem = e.upper_bound - e.current_flow
                if rem < bottleneck:
                    bottleneck = rem
                v = e.reverse.target

            # Pushing flow equal to bottleneck along the same path
            v = sink_idx
            while v != source_idx:
                e = parent_edge[v]
                e.current_flow += bottleneck
                e.reverse.current_flow -= bottleneck
                v = e.reverse.target

            total_flow += bottleneck
            parent_edge = [None] * len(self.vertices)

        return total_flow
    """
    Reference for ford_fulkerson function:
    
    GeeksforGeeks. (2023, June 1). FordFulkerson Algorithm for maximum Flow Problem. GeeksforGeeks. 
        https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
    """

def crowdedCampus(n, m, timePreferences, proposedClasses, minimumSatisfaction):
    """
    Function Description:

    This function assigns each of n students to exactly one of m classes, with each class having a fixed timeslot and a
    minimum and maximum capacity, ensuring every class meets its minimum and as many students as possible are placed in 
    one of their top 5 preferred timeslots then returns a list of class IDs or None if no valid assignment exists

    Approach Description:
    1. The function calls 'build_graph_helper' with the inputs for step 1 as this is to create the source node student nodes 
       timeslot nodes class nodes and sink node while recording all edges that enforce each class minimum and maximum capacities

    2. Every recorded timeslot to class edge and class to sink edge has its lower bound and upper bound values swapped 
       so that running one maximum flow calculation forces each class minimum capacity to be filled

    3. The function runs Ford Fulkerson to satisfy all class minimums and after this pass, it collects any timeslot to class 
       edges that still have unused capacity into a leftovers queue and collects all students whose source edge flow remains 
       zero into an unassigned students queue

    4. If the total number of leftover spots exceeds the count of unassigned students the function, it returns None. Else, 
       for each leftover spot it assigns an unassigned student by adding a capacity one edge from that student to the 
       appropriate timeslot and runs Ford Fulkerson again so each of those forced edges carries flow into the sink

    5. All bounds on the original timeslot class and class sink edges are restored to their minimum and maximum values. 
       Then for each student the function ensures a capacity one edge exists to every timeslot not yet reached so that 
       every student can access all timeslots

    6. The function runs Ford Fulkerson a third times to push as much flow as possible through each student original 
       top 5 preference edges while maximizing the number of students in one of their top 5 choices

    7. Finally for each student the function finds the one outgoing edge with flow equal to one to identify the assigned 
       timeslot then finds the outgoing edge from that timeslot with positive flow to identify the class then it counts how 
       many students ended up in one of their top 5 choices and if that count is below the required minimum the function 
       returns None else it returns the list of class IDs

    Input:
        n                   = the number of students
        m                   = the number of classes
        timePreferences     = the list of n lists, each with 5 preferred timeslot IDs per student
        proposedClasses     = the list of m triples of slot_id, min_cap, and max_cap for each class
        minimumSatisfaction = the required number of students in top 5 timeslots

    Output:
        A list of length n: allocation[i] = the assigned class ID for student i, or None if no feasible allocation exists

    Time Complexity:

        n = the number of students
        m = the number of classes
        c = the number of timeslots

        Worst Case: O(n^2)
        Analysis:
            - First, the function builds a flow network with one source, n student nodes, 20 timeslot nodes, m class nodes, 
              and one sink. Adding edges from source to students, from students to their top 5 timeslots, and from timeslots 
              to classes to sink takes O(n + m) time total.
            - Next, the code performs three separate Ford Fulkerson runs while each run finds augmenting paths by BFS. 
              In the worst case, each student sends one unit of flow, so the total flow is O(n). 
              Then the BFS explores O(n + m) edges,hence each maximum flow call takes O(n x (n + m)) time.
            - The first run pushes flow until every class meets its minimum and the second run pushes any forced assigns that 
              were added when some classes still had open spots. The third run, after restoring class capacities to their maximum, 
              pushes any remaining flow to fill classes up to their max. Each of these three runs is O(n x (n + m)), so together 
              they add up to O(n x (n + m)).
            - All other steps swapping lower and upper bounds on critical edges, scanning timeslot edges to identify leftover seats, 
              scanning student edges to find unassigned students, adding forced student→timeslot edges, and restoring bounds—each 
              require a single pass over O(n + m) items so these steps total O(n + m) and do not change the dominant cost.
            - Finally, building the allocation and checking minimum satisfaction involves scanning each student edges (O(20)per student) 
              and then scanning up to m classes per student to assign them so that takes O(n + n x m) = O(n x m), 
              which is also lower order compared to O(n x (n + m)).
            
            Collapsed Complexity:  
                - The brief states that M is a positive integer. In the extreme case where every class holds exactly one student,
                  M equals N. In every other situation M is smaller than N, so any complexity terms containing N dominate those 
                  that contain M.
                - If the number of classes grows in proportion to the number of students, we write m = O(n)
                  In other words, there is some constant c > 0 so that m less than or equal c times n when n is large
                - Then n + m less than or equal n + c x n = (1 + c) x n. That expression is still on the order of n
                - Therefore, n x (n + m) less than or equal n x ((1 + c) x n) = (1 + c) x n^2, which is on the order of n^2
                - In a nutshell, when m = O(n), O(n x (n + m)) collapses to O(n x 2n) = O(n^2)

    Auxiliary Space Complexity: 

        Worst Case: O(n + m)
        Analysis:
            - The main data structures stored throughout the function are the flow network vertices and edges
            - There is one source, n student nodes, 20 timeslot nodes which is a constant, m class nodes, and one sink, 
              totalling O(n + m) vertices so the number of edges is also O(n + m) so together, storing all vertices and edges 
              requires O(n + m) space
            - Then, the Ford Fulkerson implementation uses a BFS queue and a parent array of size equal to the 
              number of vertices. Both of those require O(n + m) space for each run and since the same queue and parent array 
              are reused for each of the three runs, the peak space for these is still O(n + m)
            - The leftover deque can contain at most one entry per class with open spots, which is O(m) while the 
              unassigned_students deque can contain at most n entries, which is O(n) and both deques together require O(n + m) space
            - Storing helper arrays uses O(n + m) space

        Collapsed Complexity:
            - Since the number of classes m is a positive integer that grows at most proportionally to n (m = O(n)) as mentioned 
              in the brief, every O(n + m) becomes O(n), hence the total space usage collapses to O(n)

    Input Complexity:
        Size: O(n + m)
        Analysis:
            - The timePreferences input is a list of n lists, each of length c which is 20, so it gives O(n x c) = O(n)
            - The proposedClasses input is a list of m entries, each containing 3 integers, so it gives O(m)
    """
    # Building the flow network and all required vertices and edges for the graph
    source, all_vertices, students, timeslot_nodes, sink, critical_edges = \
        build_graph_helper(n, m, timePreferences, proposedClasses)

    # Swapping bounds so minimum requirements become capacities for first ford fulkerson algorithm run
    # Then we impose that each class minimum spots are filled first
    for edge in critical_edges:
        edge.upper_bound, edge.lower_bound = edge.lower_bound, edge.upper_bound

    # Initializing the flow network object 
    network = FlowNetwork(all_vertices)
    # Pushing as much flow as possible to satisfy minimum requirements for all classes through ford fulkerson algorithm
    network.ford_fulkerson(source.idx, sink.idx)

    # Here, this is to track any remaining open spots in classes like edges not full after first pass
    leftovers = deque()
    leftover_count = 0
    for idx in range(len(timeslot_nodes)):
        timeslot_v = timeslot_nodes[idx]
        if timeslot_v is None:
            continue
 
        # Each timeslot has at most one outgoing edge to its class vertex
        for edge in timeslot_v.edges:
            remaining = edge.upper_bound - edge.current_flow
            if remaining > 0:
                leftovers.append(edge)
                leftover_count += remaining

    # Finding students who did not get assigned in the first flow pass
    unassigned_students = deque()
    for i in range(len(source.edges)):
        edge = source.edges[i]
        if edge.current_flow == 0:
            student_vertex = all_vertices[edge.target]
            unassigned_students.append(student_vertex)

    # When more required minimum spots than unassigned students, no solution is available in this case
    if leftover_count > len(unassigned_students):
        return None

    # For each leftover spot, forcing one unassigned student into that timeslot
    for i in range(leftover_count):
        student_v = unassigned_students.popleft()
        left_edge = leftovers[0]

        # Adding edge: student to timeslot with capacity = 1
        forced_edge = Edge(source=student_v.idx, target=left_edge.source, lower_bound=0, upper_bound=1)
        rev_forced_edge = Edge(source=left_edge.source, target=student_v.idx, lower_bound=0, upper_bound=0)
        forced_edge.reverse = rev_forced_edge
        rev_forced_edge.reverse = forced_edge
        student_v.add_edge(forced_edge)

        # If the spot is now filled, remove it from leftovers
        if (left_edge.upper_bound - left_edge.current_flow) == 0:
            leftovers.popleft()

    # Running ford fulkerson algorithm again to push forced assignments into the network
    network.ford_fulkerson(source.idx, sink.idx)

    # Restoring original lower or upper bounds on all class related edges
    for edge in critical_edges:
        edge.lower_bound, edge.upper_bound = edge.upper_bound, edge.lower_bound

    # Ensuring every student has an edge to every timeslot
    for student_v in students:
        for slot_id in range(20):
            timeslot_v = timeslot_nodes[slot_id]
            if not timeslot_v:
                continue

            # Checking if this student already has an edge to slot_id
            already_connected = False
            for edge in student_v.edges:
                if all_vertices[edge.target].data == slot_id:
                    already_connected = True
                    break

            # If no edge exists, create a new edge plus its reverse edge
            if not already_connected:
                new_edge = Edge(source=student_v.idx, target=timeslot_v.idx, lower_bound=0, upper_bound=1)
                rev_new_edge = Edge(source=timeslot_v.idx, target=student_v.idx, lower_bound=0, upper_bound=0)
                new_edge.reverse = rev_new_edge
                rev_new_edge.reverse = new_edge
                student_v.add_edge(new_edge)

    # Running ford fulkerson algorithm the last time to maximize number of students in their original top 5 timeslots
    network.ford_fulkerson(source.idx, sink.idx)

    # In here, we need to find which timeslot the students were assigned to
    allocation = [None] * n
    for i in range(n):
        student_v = students[i]
        # Looking through every edge that leaves this student
        for edge in student_v.edges:
            # Then, there will be exactly one edge out of the student should carry flow that is 1
            if edge.current_flow == 1:
                allocation[i] = all_vertices[edge.target]  
                break
        # If no edge with flow = 1 is found, this student is unplaced
        if allocation[i] is None:
            return None

    # For each assigned timeslot, finding which class was assigned through that timeslot
    for i in range(n):
        timeslot_v = allocation[i]
        for edge in timeslot_v.edges:
            if edge.current_flow > 0:
                allocation[i] = all_vertices[edge.target].data 
                # Decrementing so next student do not reuse same edge
                edge.current_flow -= 1 
                break

    # Counting how many students received a top 5 timeslot preference
    sat_count = 0
    for i in range(n):
        class_id = allocation[i]
        slot_id = proposedClasses[class_id][0] 
        for k in range(5):
            if slot_id == timePreferences[i][k]:
                sat_count += 1
                break

    # To check if it meets the preference threshold
    if sat_count < minimumSatisfaction:
        return None

    return allocation

# Question 2
# Typo
class TrieNode:
    __slots__ = ("children", "word_index", "min_index_in_subtree")

    def __init__(self):
        """
        Function Description:
        
        This function creates a node for a prefix trie that handles lowercase letters from a to z

        Approach Description:
        1. Allocating a list children of length 26, one slot for each letter from a to z
        2. Setting the word_index to -1, meaning that no word ends at this node
        3. Setting the min_index_in_subtree to None, meaning that no word index recorded yet

        Time Complexity:
            - O(1), since it does a fixed amount of work

        Auxiliary Space Complexity:
            - O(1), since it stores only three small fields
            Three fields:
            children: This field has fixed-size list of 26 slots where each slot corresponds to a letter from a to z and holds 
                      either a link to the next node for that letter or None
            word_index: This field represents an integer that is -1 by default and if a complete word ends at this node, 
                        it then stores that word index, else, it stays -1
            min_index_in_subtree: This field holds the smallest index of any word stored in this node entire subtree where 
                                  it starts as None and is updated whenever a word is inserted beneath this node
        """
        # children is a fixed-size list of length 26 for each lowercase letterfrom a to z
        self.children = [None] * 26
        # word_index will be storing the index of a complete word ending at this node, or -1 if none
        self.word_index = -1
        # min_index_in_subtree tracks the smallest index of any word in this node subtree
        self.min_index_in_subtree = None


class Bad_AI:
    def __init__(self, list_words):
        """
        Function Description:

        This function builds a set of tries where each trie holds all words of the same length from list_words where 
        each node tracks the smallest index of any word in its subtree so that future searches can be faster

        Approach Description:
        1. Storing the reference to list_words so we can use it in searches
        2. Finding max_word_length by scanning the length of each word once
        3. Creating a self.tries, a list of size where each slot will hold either None or the root of a trie 
           for words of that length
        4. For each word at index idx in list_words:
           - Let length = len(word) and if self.tries[length] is None, then create a new root node
           - At root, it updates the min_index_in_subtree to idx if it is None or larger than idx
           - Then, traverse each character in word to compute child_index = ord(char) - ord("a") and if there is 
             no child at children[child_index],it creates a new node then mmove node to that child, and update 
             its min_index_in_subtree to idx if needed
        5. After the last character, set the node.word_index = idx to mark the end of that word

        Input:
            list_words = the list of words with all lowercase string

        Output:
            None

        Time Complexity:

            N = the number of words
            C = the total number of characters across all words

            Worst Case: O(C)
            Analysis:
            - Finding the longest word takes O(N) by scanning the length of each word
            - Inserting all words into tries visits each character once, so it is O(C)
            - Updating min_index_in_subtree and word_index is constant work per node visit

        Auxiliary Space Complexity:

            Worst Case: O(C)
            Expanded Details:
            1. The tries list itself uses space proportional to max_word_length + 1, which is less than or equal to C, having it O(C)
            2. Each character in every word creates at most one TrieNode where each node includes:
               - the list of 26 child references
               - the integer word_index
               - the integer reference for min_index_in_subtree
               where all nodes together require O(C) space
            3. No other temporary structures are created during insertion beyond local variables

        Input Complexity:
            Size: O(N + C)
            Analysis:
            - list_words holds N words and C characters total
        """
        # Storing the original list of words
        self.words = list_words

        # Finding the maximum word length among all words
        max_word_length = 0
        for word in list_words:
            word_length = len(word)
            if word_length > max_word_length:
                max_word_length = word_length

        # Then, creating an array of size max_word_length + 1 for separate tries
        self.tries = [None] * (max_word_length + 1)

        # Inserting each word into the trie corresponding to its length
        for idx, word in enumerate(list_words):
            length = len(word)
            # When there is no trie root of this length, create one for it
            if self.tries[length] is None:
                self.tries[length] = TrieNode()
            node = self.tries[length]

            # Updating root min_index_in_subtree if this word’s index is smaller
            existing_min = node.min_index_in_subtree
            if existing_min is None or idx < existing_min:
                node.min_index_in_subtree = idx

            # Walking or creating child nodes for each character
            for char in word:
                # Computing child_index based on the position in the alphabet
                child_index = ord(char) - ord('a')
                child_node = node.children[child_index]
                if child_node is None:
                    child_node = TrieNode()
                    node.children[child_index] = child_node
                node = child_node
                # Update this node min_index_in_subtree if needed
                existing_min = node.min_index_in_subtree
                if existing_min is None or idx < existing_min:
                    node.min_index_in_subtree = idx

            # Marking the end of the word with its index
            node.word_index = idx

    def check_word(self, sus_word):
        """
        Function Description:

        This function returns a list of all words from the original list that have the same length as sus_word
        and differ by exactly one character only substitutions allowed where it ignores words of other lengths

        Approach Description:
        1. We are first setting J as the len(sus_word) and when J is outside the range of self.tries or self.tries[J] is None, 
           return [].
        2. Performing a partial trie traversal to skip exact match prefixes by setting node = self.tries[J] and index = 0 while 
           index is less than J, compute the char_code = ord(sus_word[index]) - ord('a') and if node.children[char_code] is None, 
           break out of the loop, else, set node as node.children[char_code] and index += 1
        3. Perform a full linear scan over each word w in self.words by checking len(w) not equal to J and skip to the next word, 
           else, compare w and sus_word character by character, counting mismatches and if mismatches == 1, then append w to the 
           result list
        4. Return the result list containing all words with exactly 1 by substitution

        Input:
            sus_word = the word to check as a string

        Output:
            A list of words from the original list that match the criteria

        Time Complexity:

            N = the number of words in self.words
            J = the length of sus_word
            K = the number of matched words
            X = the total number of characters returned in the correct result

            Worst Case: O(J x N) + O(X)
            Analysis:
            - Exact match trie traversal takes a total of J steps, where each performing one array lookup is O(J)
            - Linear scan over N words and:
              - For each word that the length is not equal to J, the length check takes a constant time
              - For each word of length J, comparing up to J characters and stopping early if mismatches exceed one takes O(J)
              - In the worst case, each of the N words differs in exactly one character at the last position, 
              requiring J comparisons each is O(J x N)
            - Then, appending K matched words of length J into the result list takes up K x J operations, 
              which is X where X is the total number of characters returned in the correct result
            - So, the worst case total time now adds up to O(J x N) + O(X)

        Auxiliary Space Complexity:
            - Exact match trie traversal uses only a constant number of pointers and counters, so that
              step requires only O(1)
            - Scanning each dictionary word uses a mismatch counter and loop index, occupying O(1)
            - The only non constant storage is the result list, which contains K words of length J,
              for a total of X characters
            - No additional arrays or recursion stacks of size proportional to N or J are used, hence, the auxiliary space is O(X)

        Input Complexity:
            Size: O(C + J)
            Analysis:
            - list_words have N words whose total characters sum to C, so storing or reading this list is O(C) 
              while sus_word is a single string of length J, sit then needs O(J) space.
            - Because these two inputs coexist in memory during execution, so the combined version becomes O(C + J)
        """
        # Getting length of sus_word
        J = len(sus_word)

        # When J is outside the tries list or there is no trie for this length, return empty
        if J >= len(self.tries) or self.tries[J] is None:
            return []

        # Traversing the trie as far as characters match
        node = self.tries[J]
        index = 0
        while index < J:
            # Computing the index for this character
            char_code = ord(sus_word[index]) - ord('a')
            next_node = node.children[char_code]
            if next_node is None:
                break
            node = next_node
            index += 1

        result = []
        # Scanning every word in the original list
        for w in self.words:
            # Skipping words whose length not equal to J
            if len(w) != J:
                continue

            # Comparing characters to count mismatches
            mismatches = 0
            for i in range(J):
                if w[i] != sus_word[i]:
                    mismatches += 1
                    if mismatches > 1:
                        break

            # When mismatches = 1, add word to the result
            if mismatches == 1:
                result.append(w)

        return result

