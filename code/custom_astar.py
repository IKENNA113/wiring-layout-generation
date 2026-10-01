import cv2
import numpy as np
import os
import json
import math
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import heapq

SCALE = 10  # grid cell size in pixels (s = 10 px in the paper)
# WIDTH =  11520//SCALE
# HEIGHT = 8320//SCALE
WIDTH =  11520
HEIGHT = 8320


#drawing vars
# Global variables
drawing = False  # True if mouse is pressed
points = []  # List to store polygon points
filled_polygons = []  # List to store filled polygons
color = (0, 165, 255)  # White color

def drawSegmentationMap(inputImage):
    global drawing, points, filled_polygons, color

    # Mouse callback function
    def draw_shape(event, x, y, flags, param):
        global points, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fill_polygon(img, points)
            points = []  # Clear points list when mouse button is released

    # Function to draw polygon on the image
    def draw_polygon(img, points):
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(img, points[i], points[i + 1], color, 2)

    # Function to fill the polygon with white color
    def fill_polygon(img, points):
        if len(points) > 2:
            pts = np.array(points, np.int32)
            cv2.fillPoly(img, [pts], color)
            filled_polygons.append(pts)  # Store filled polygon

    # Function to erase the last filled polygon
    def erase_last_polygon():
        global img
        if filled_polygons:
            temp_img = original_img.copy()  # Restore original image
            for polygon in filled_polygons[:-1]:
                cv2.fillPoly(temp_img, [polygon], color)
            img = temp_img
            filled_polygons.pop()  # Remove the last filled polygon


    # Function to save the resulting image
    def save_image(filename, img):
        # filename = input("Enter the filename to save (e.g., image.jpg): ")
        os.makedirs('segmentation_map', exist_ok=True)
        savingImage = np.zeros_like(img)
        for polygon in filled_polygons:
            cv2.fillPoly(savingImage, [polygon], (255, 255, 255))  # Fill filled polygons with white on the mask
        #convert the image to original size
        # savingImage  = np.ones((actual_img.shape[0], actual_img.shape[1], 3), dtype=np.uint8) * 255
        savingImage = cv2.resize(savingImage, (actual_img.shape[1], actual_img.shape[0]), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.getcwd() + "/output/" + file[:-4] + "_obstacle.png", savingImage)
        print(f"Image saved as {filename}")



    def fill_background():
        global img
        mask = np.zeros_like(img)  # Create a black mask
        for polygon in filled_polygons:
            cv2.fillPoly(mask, [polygon], (255, 255, 255))  # Fill filled polygons with white on the mask
        img = cv2.bitwise_and(img, mask)  # Apply mask to the original image


        
    image_path = inputImage
    # get the images ends with .jpg or .png
    print("Working on image: ", image_path)
    if not image_path.endswith('.jpg') and not image_path.endswith('.png'):
        return False

    # Load the image
    img = cv2.imread(image_path)
    dim = (WIDTH, HEIGHT)



    # Resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    actual_img = img.copy()
    #resize the image with same aspect ratio
    scale_percent = 10 # percent of original size

    # width = int(img.shape[1] * scale_percent / 100)
    # height = int(img.shape[0] * scale_percent / 100)
    # dim = (width, height)
    


    if img is None:
        print("Error: Unable to load image.")
        exit()

    original_img = img.copy()  # Store the original image

    ## cv2.namedWindow('image')
    #cv2.setMouseCallback('image', draw_shape)
    
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw_shape)

    while True:
        temp_img = np.copy(img)

        if len(points) > 0:
            draw_polygon(temp_img, points)

        for polygon in filled_polygons:
            cv2.fillPoly(temp_img, [polygon], color)

        cv2.imshow('image', temp_img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('e'):  # Press 'e' to erase the last filled polygon
            erase_last_polygon()
        elif k == ord('r'):  # Press 'r' to reset the image
            img = original_img.copy()  # Restore original image
            points = []
            filled_polygons = []
        elif k == ord('c'):  # Press 'c' to clear the polygon points
            points = []
            temp_img = np.copy(img)
        # Inside the while loop
        elif k == ord('b'):  # Press 'b' to fill the background with black color except filled polygons
            fill_background()

        elif k == ord('s'):  # Press 's' to save the resulting image
            save_image(image, img)
        elif k == 27:  # ESC key to exit
            img = original_img.copy()  # Restore original image
            points = []
            filled_polygons = []
            break

    cv2.destroyAllWindows()

def draw_filled_rectangle_with_opacity(image, top_left, bottom_right, color, opacity):

    # Create a copy of the original image to avoid modifying the original
    overlay = image.copy()

    # Draw the filled rectangle on the overlay image
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)

    # Add the overlay with opacity
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

    return image


#class imports
class Component:
    
    def __init__(self, name, location,routeId = -1,category=0):
        self.name = name
        self.location = location
        self.centerPoint = (int((location[0]+location[2])/2), int((location[1]+location[3])/2))
        self.edgePointTop = None
        self.edgePointBottom = None
        self.edgePointLeft = None
        self.edgePointRight = None
        self.routeId = routeId
        self.calEdgePoints()
        # Curated bright palette — high saturation, high luminance, distinguishable
        # against typical CAD-diagram backgrounds. Cycled by routeId so all
        # components on the same circuit share a colour and different circuits
        # are visually distinct. Replaces random RGB which often produced muddy,
        # low-contrast colours on light backgrounds.
        # OpenCV uses BGR ordering — these tuples are (B, G, R).
        BRIGHT_PALETTE = [
            (255,   0,   0),   # vivid blue
            (0,     0, 255),   # vivid red
            (0,   200,   0),   # vivid green
            (255,   0, 255),   # magenta
            (0,   165, 255),   # orange
            (255, 255,   0),   # cyan
            (128,   0, 255),   # pink-red
            (0,   255, 128),   # spring green
            (255, 128,   0),   # azure
            (128, 255,   0),   # lime
            (255,   0, 128),   # purple
            (0,   128, 255),   # tangerine
            (255, 192,   0),   # turquoise
            (192,   0, 255),   # rose
            (0,   255, 255),   # yellow
            (128, 128, 255),   # salmon
        ]
        idx = max(0, int(routeId)) % len(BRIGHT_PALETTE)
        self.color = BRIGHT_PALETTE[idx]
        self.pannelConnection = False
        self.category = category
        self.presenceOFCategory2 = False
        self.distanceFromPannelBoardCenter = 0
        self.usedPoints = []

    def __str__(self):
        return f"{self.name} - {self.description} - {self.price}"
    
    def calEdgePoints(self):
        #calculate the center edge point
        self.edgePointTop = (int((self.location[0]+self.location[2])/2), self.location[1])
        self.edgePointBottom = (int((self.location[0]+self.location[2])/2), self.location[3])
        self.edgePointLeft = (self.location[0], int((self.location[1]+self.location[3])/2))
        self.edgePointRight = (self.location[2], int((self.location[1]+self.location[3])/2))
        return True

class Router:
    def __init__(self,Obmap = None):
        self.routes = {}
        self.plainX = 11520
        self.plainY = 8320
        self.grid = self.createGrid()
        self.obstacleImage = None
        #create the map
        self.map = Map(self.plainX, self.plainY)
        self.map.setObstacleMap(Obmap)
        self.map.create_route_map(self.plainY, self.plainX)
        self.pathFindingArray = self.map.getRouteMap()

    #create a grid image with plain dimension with 30x30 grid
    def createGrid(self):
        grid = np.zeros((self.plainY, self.plainX, 3), dtype=np.uint8)
        #create a grid
        for i in range(0, self.plainY, 30):
            for j in range(0, self.plainX, 30):

                #set white color
                grid[i:i+30, j:j+30] = [0, 255, 0]
        return grid

    def buildGraphForCluster(self, tier2Component, tier3Components):
        """
        Build a fully connected weighted graph for a Tier 2 cluster.
        Nodes: Tier 2 (root) + all Tier 3 components
        Edges: All possible connections with distances as weights
        """
        G = nx.Graph()
        
        # Add Tier 2 as root node
        tier2NodeId = f"T2_{tier2Component.name}_{tier2Component.centerPoint}"
        G.add_node(tier2NodeId, 
                   pos=tier2Component.centerPoint, 
                   component=tier2Component,
                   tier=2)
        
        # Add all Tier 3 nodes
        for idx, tier3 in enumerate(tier3Components):
            tier3NodeId = f"T3_{idx}_{tier3.name}_{tier3.centerPoint}"
            G.add_node(tier3NodeId, 
                       pos=tier3.centerPoint, 
                       component=tier3,
                       tier=3)
        
        # Create fully connected graph with distance weights
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                comp1 = G.nodes[nodes[i]]['component']
                comp2 = G.nodes[nodes[j]]['component']
                distance = self.distanceCalculate(comp1.centerPoint, comp2.centerPoint)
                G.add_edge(nodes[i], nodes[j], weight=distance)
        
        return G, tier2NodeId

    def dfsMinimalPath(self, G, tier2Node, tier3Nodes):
        """
        Standard TSP solver focused on minimizing TOTAL path distance.
        
        Combines:
        1. Nearest Neighbor initialization
        2. 2-opt improvement
        3. Or-opt (relocate sequences of 1-3 nodes)
        4. Simulated Annealing with total distance objective
        """
        if len(tier3Nodes) == 0:
            return [tier2Node]
        
        import random
        import math
        
        def get_edge(p, i):
            """Get edge weight between p[i] and p[i+1]"""
            if i < 0 or i >= len(p) - 1:
                return 0
            return G[p[i]][p[i+1]]['weight']
        
        def path_distance(p):
            """Get total path distance"""
            return sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1))
        
        # Step 1: Nearest Neighbor initialization
        path = [tier2Node]
        remaining = set(tier3Nodes)
        current = tier2Node
        
        while remaining:
            nearest = min(remaining, key=lambda n: G[current][n]['weight'])
            path.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        print(f"  After Nearest Neighbor: total={path_distance(path):.2f}")
        
        # Step 2: 2-opt improvement (focus on total distance)
        improved = True
        while improved:
            improved = False
            current_dist = path_distance(path)
            
            for i in range(1, len(path) - 2):
                for j in range(i + 2, len(path)):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_dist = path_distance(new_path)
                    
                    if new_dist < current_dist - 0.01:
                        path = new_path
                        current_dist = new_dist
                        improved = True
        
        print(f"  After 2-opt: total={path_distance(path):.2f}")
        
        # Step 3: Or-opt (relocate sequences of 1, 2, 3 nodes)
        improved = True
        while improved:
            improved = False
            current_dist = path_distance(path)
            
            for seq_len in [1, 2, 3]:  # Try moving 1, 2, or 3 consecutive nodes
                for i in range(1, len(path) - seq_len):  # Start position (skip tier2)
                    # Extract sequence
                    sequence = path[i:i+seq_len]
                    rest = path[:i] + path[i+seq_len:]
                    
                    # Try inserting sequence at each position
                    for j in range(1, len(rest)):  # Insert positions (skip position 0)
                        new_path = rest[:j] + sequence + rest[j:]
                        new_dist = path_distance(new_path)
                        
                        if new_dist < current_dist - 0.01:
                            path = new_path
                            current_dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        print(f"  After Or-opt: total={path_distance(path):.2f}")
        
        # Step 4: Simulated Annealing (total distance objective)
        best_path = path[:]
        best_dist = path_distance(path)
        current_path = path[:]
        current_dist = best_dist
        
        # SA parameters
        T = 1000.0  # Initial temperature
        T_min = 0.1
        alpha = 0.995  # Cooling rate
        iterations_per_temp = 100
        
        sa_iterations = 0
        
        # Only run SA if path allows meaningful moves (needs at least 4 nodes for 2-opt logic below)
        if len(path) >= 4:
            while T > T_min:
                for _ in range(iterations_per_temp):
                    sa_iterations += 1
                    
                    # Generate neighbor (random 2-opt or Or-opt move)
                    move_type = random.choice(['2opt', 'oropt'])
                    
                    if move_type == '2opt':
                        try:
                            i = random.randint(1, len(current_path) - 3)
                            j = random.randint(i + 2, len(current_path) - 1)
                            new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                        except ValueError:
                            continue
                    else:  # oropt
                        seq_len = random.randint(1, 3)
                        if len(current_path) - seq_len <= 2:
                            continue
                        try:
                            i = random.randint(1, len(current_path) - seq_len - 1)
                            sequence = current_path[i:i+seq_len]
                            rest = current_path[:i] + current_path[i+seq_len:]
                            j = random.randint(1, len(rest) - 1)
                            new_path = rest[:j] + sequence + rest[j:]
                        except ValueError:
                            continue
                    
                    new_dist = path_distance(new_path)
                    
                    # Accept or reject
                    delta = new_dist - current_dist
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_path = new_path
                        current_dist = new_dist
                        
                        if current_dist < best_dist:
                            best_path = current_path[:]
                            best_dist = current_dist
                
                T *= alpha
        
        path = best_path
        print(f"  After SA ({sa_iterations} iterations): total={path_distance(path):.2f}")
        
        # Step 5: Final 2-opt cleanup
        improved = True
        while improved:
            improved = False
            current_dist = path_distance(path)
            
            for i in range(1, len(path) - 2):
                for j in range(i + 2, len(path)):
                    new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                    new_dist = path_distance(new_path)
                    
                    if new_dist < current_dist - 0.01:
                        path = new_path
                        current_dist = new_dist
                        improved = True
        
        # Calculate final stats
        total_path_distance = path_distance(path)
        
        print(f"Standard TSP (NN + 2-opt + Or-opt + SA):")
        print(f"  Path length: {len(path)} nodes")
        print(f"  Sequential path distance: {total_path_distance:.2f} pixels")
        
        return path

    def _creates_cycle(self, edges, new_node1, new_node2, all_nodes):
        """Check if adding an edge creates a cycle (excluding complete path)"""
        if len(edges) == 0:
            return False
        
        # Build adjacency list from existing edges
        adj = {node: [] for node in all_nodes}
        for n1, n2 in edges:
            adj[n1].append(n2)
            adj[n2].append(n1)
        
        # DFS to check if new_node1 and new_node2 are already connected
        visited = set()
        stack = [new_node1]
        
        while stack:
            current = stack.pop()
            if current == new_node2:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(adj[current])
        
        return False

    def _edges_to_path(self, edges, start_node):
        """Convert list of edges to ordered path starting from start_node"""
        if len(edges) == 0:
            return [start_node]
        
        # Build adjacency list
        adj = {}
        for n1, n2 in edges:
            if n1 not in adj:
                adj[n1] = []
            if n2 not in adj:
                adj[n2] = []
            adj[n1].append(n2)
            adj[n2].append(n1)
        
        # Traverse from start_node
        path = [start_node]
        current = start_node
        visited = {start_node}
        
        while len(path) < len(adj):
            # Find unvisited neighbor
            for neighbor in adj[current]:
                if neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)
                    current = neighbor
                    break
    
        return path

    def _get_max_edge_weight(self, G, path):
        """Get the maximum edge weight in the path"""
        if len(path) < 2:
            return 0
        return max(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

    def visualizeGraph(self, G, path, clusterName, routingIds, clusterId):
        """
        Visualize the graph and save as image + text log
        Uses routing IDs (e.g., 6_0_1, 6_0_2) for node naming
        """
        # Create routing ID mapping for nodes
        node_routing_ids = {}
        node_routing_ids[path[0]] = f"{routingIds}_{clusterId}"  # Tier 2 node
        
        for idx, node in enumerate(path[1:], start=1):  # Tier 3 nodes
            node_routing_ids[node] = f"{routingIds}_{clusterId}_{idx}"
        
        # === TEXT LOGGING ===
        log_text = []
        log_text.append(f"{'='*60}")
        log_text.append(f"GRAPH ANALYSIS - Cluster {clusterName}")
        log_text.append(f"{'='*60}\n")
        
        # Graph Statistics
        log_text.append(f"Graph Statistics:")
        log_text.append(f"  Total Nodes: {len(G.nodes())}")
        log_text.append(f"  Total Edges: {len(G.edges())}")
        log_text.append(f"  Tier 2 Nodes: {len([n for n in G.nodes() if G.nodes[n]['tier'] == 2])}")
        log_text.append(f"  Tier 3 Nodes: {len([n for n in G.nodes() if G.nodes[n]['tier'] == 3])}\n")
        
        # Path Information
        log_text.append(f"Minimax TSP Optimal Path:")
        log_text.append(f"  Path Length: {len(path)} nodes")
        total_path_distance = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
        log_text.append(f"  Total Path Distance: {total_path_distance:.2f} pixels\n")
        
        # Node Details with Routing IDs
        log_text.append(f"Path Sequence:")
        for idx, node in enumerate(path):
            comp = G.nodes[node]['component']
            tier = G.nodes[node]['tier']
            pos = G.nodes[node]['pos']
            routing_id = node_routing_ids[node]
            
            if idx == 0:
                log_text.append(f"  [{routing_id}] {comp.name} at {pos} (START - Tier {tier})")
            elif idx == len(path) - 1:
                prev_node = path[idx-1]
                edge_weight = G[prev_node][node]['weight']
                log_text.append(f"  [{routing_id}] {comp.name} at {pos} (END - Tier {tier}) - Distance from prev: {edge_weight:.2f}")
            else:
                prev_node = path[idx-1]
                edge_weight = G[prev_node][node]['weight']
                log_text.append(f"  [{routing_id}] {comp.name} at {pos} (Tier {tier}) - Distance from prev: {edge_weight:.2f}")
        
        log_text.append(f"\n{'='*60}")
        log_text.append(f"Edge Details (Path Connections):")
        log_text.append(f"{'='*60}")
        
        for i in range(len(path)-1):
            node1 = path[i]
            node2 = path[i+1]
            comp1 = G.nodes[node1]['component']
            comp2 = G.nodes[node2]['component']
            weight = G[node1][node2]['weight']
            routing_id1 = node_routing_ids[node1]
            routing_id2 = node_routing_ids[node2]
            
            log_text.append(f"  Connection {i+1}: [{routing_id1}] → [{routing_id2}]")
            log_text.append(f"    {comp1.name} → {comp2.name}")
            log_text.append(f"    From: {comp1.centerPoint}")
            log_text.append(f"    To:   {comp2.centerPoint}")
            log_text.append(f"    Distance: {weight:.2f} pixels\n")
        
        # MST Statistics
        mst = nx.minimum_spanning_tree(G, weight='weight')
        mst_total_weight = sum(mst[u][v]['weight'] for u, v in mst.edges())
        log_text.append(f"\nMinimum Spanning Tree (MST) Info:")
        log_text.append(f"  MST Total Weight: {mst_total_weight:.2f} pixels")
        log_text.append(f"  MST Edges: {len(mst.edges())}")
        log_text.append(f"  Path vs MST Efficiency: {(mst_total_weight/total_path_distance)*100:.2f}%\n")
        
        # All Edges Information (Full Graph) with Routing IDs
        log_text.append(f"\n{'='*60}")
        log_text.append(f"Complete Graph - All Possible Connections:")
        log_text.append(f"{'='*60}")
        
        edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
        for idx, (node1, node2, data) in enumerate(edges_sorted, 1):
            comp1 = G.nodes[node1]['component']
            comp2 = G.nodes[node2]['component']
            weight = data['weight']
            
            # Get routing IDs if available
            id1 = node_routing_ids.get(node1, "N/A")
            id2 = node_routing_ids.get(node2, "N/A")
            
            in_path = (node1, node2) in [(path[i], path[i+1]) for i in range(len(path)-1)]
            in_path = in_path or (node2, node1) in [(path[i], path[i+1]) for i in range(len(path)-1)]
            in_path_marker = " ✓ [USED IN PATH]" if in_path else ""
            
            log_text.append(f"  {idx}. [{id1}] ↔ [{id2}]: {weight:.2f} pixels{in_path_marker}")
            log_text.append(f"      ({comp1.name} ↔ {comp2.name})")
        
        log_text.append(f"\n{'='*60}\n");
        
        # Save text log
        os.makedirs('output/graphs', exist_ok=True)
        log_filename = f'output/graphs/graph_{clusterName}_log.txt'
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(log_text))
        print(f"Graph log saved: {log_filename}")
        
        # === VISUALIZATION ===
        plt.figure(figsize=(16, 12))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw all edges (gray, thin)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.4)
        
        # Draw path edges (red, thick)
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, alpha=0.8)
        
        # Draw nodes
        tier2_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 2]
        tier3_nodes = [n for n in G.nodes() if G.nodes[n]['tier'] == 3]
        
        nx.draw_networkx_nodes(G, pos, nodelist=tier2_nodes, node_color='green', 
                               node_size=600, label='Tier 2', alpha=0.9)
        nx.draw_networkx_nodes(G, pos, nodelist=tier3_nodes, node_color='orange', 
                               node_size=400, label='Tier 3', alpha=0.8)
        
        # Create labels with routing IDs
        labels = {}
        for node in path:
            labels[node] = node_routing_ids[node]
        nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
        
        # Add edge weights on path
        edge_labels = {}
        for i in range(len(path)-1):
            weight = G[path[i]][path[i+1]]['weight']
            edge_labels[(path[i], path[i+1])] = f"{weight:.0f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, font_color='darkred')
        
        # Add title with statistics
        plt.title(f"Minimax TSP Routing - Cluster {clusterName}\n" + 
                 f"Nodes: {len(G.nodes())} | Path Distance: {total_path_distance:.0f} px",
                 fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=11)
        plt.axis('off')
        plt.tight_layout()
        
        # Save graph image
        plt.savefig(f'output/graphs/graph_{clusterName}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Graph saved: output/graphs/graph_{clusterName}.png")

    def routeTheGridAstar(self, componentList, image, Obmap=None):
        map = Map(self.plainX, self.plainY)
        map.setObstacleMap(Obmap)
        map.create_route_map(self.plainY, self.plainX)

        # ------------------------------------------------------------------
        # NEW LOGIC: Mark ALL components as obstacles (Blocked) on the grid
        # This prevents wires from routing OVER symbols.
        # ------------------------------------------------------------------
        print("Masking components on A* grid...")
        for comp in componentList:
            # Get component boundaries in grid coords (scale by 10)
            # Add small padding to stay clear of edges
            l, t, r, b = comp.location
            grid_l, grid_t = int(l // SCALE), int(t // SCALE)
            grid_r, grid_b = int(r // SCALE), int(b // SCALE)
            
            # Block the rectangle area
            for y in range(grid_t, grid_b + 1):
                for x in range(grid_l, grid_r + 1):
                     # Ensure within bounds
                     if 0 <= y < len(map.pathFindingArrayStar) and 0 <= x < len(map.pathFindingArrayStar[0]):
                         map.pathFindingArrayStar[y][x] = 1 # 1 = FLAGGED AS OBSTACLE
        
        # ------------------------------------------------------------------
        # NEW LOGIC: Unblock connection points
        # Ensure the specific entry/exit points are walkable (0)
        # ------------------------------------------------------------------
        for comp in componentList:
            for pt in self.getConnectionPoints(comp):
                 px, py = int(pt[0] // SCALE), int(pt[1] // SCALE)
                 
                 # Unblock a small 1x1 or 3x3 area around the point to ensure connectivity
                 # Using 1 point might be fragile if on edge
                 for dy in [-1, 0, 1]:
                     for dx in [-1, 0, 1]:
                         ny, nx = py + dy, px + dx
                         if 0 <= ny < len(map.pathFindingArrayStar) and 0 <= nx < len(map.pathFindingArrayStar[0]):
                             map.pathFindingArrayStar[ny][nx] = 0 # 0 = WALKABLE
        
        
        routeComponentList = {}
        dataWriter = ""

        for comp in componentList:
            if comp.routeId not in routeComponentList:
                routeComponentList[comp.routeId] = []
            routeComponentList[comp.routeId].append(comp)

        for routingIds in routeComponentList.keys():
            routeComponentList[routingIds] = sorted(routeComponentList[routingIds], 
                                                   key=lambda x: x.distanceFromPannelBoardCenter)
        
        completeDistance = 0
        
        for routingIds in routeComponentList.keys():
            # FILTER: Only process panelboard 6
            # if routingIds != 6:
            #     print(f"Skipping panelboard {routingIds} (only processing panelboard 6)")
            #     continue
            
            print(f"\n{'='*60}")
            print(f"Processing Panelboard {routingIds}")
            print(f"{'='*60}")
            
            routeList = []
            
            # Find panelboard
            for comp2 in componentList:
                if comp2.routeId == routingIds and comp2.name == "PANELBOARD":
                    pannelLocation = comp2.location
                    pannelCenterPoint = comp2.centerPoint
                    color = comp2.color
                    break

            routeComponentList[routingIds] = sorted(routeComponentList[routingIds], 
                                                   key=lambda x: self.distanceCalculate(x.centerPoint, pannelCenterPoint))
            
            totalBranchDistance = 0
            caegory2ClusterList = []
            total_bends = 0
            total_segments = 0
            
            # Route Category 2 (Tier 2) components
            for comp in routeComponentList[routingIds]: 
                if comp.category == 2:
                    centerPoint = comp.edgePointLeft
                    destination_point = pannelCenterPoint
                    caegory2ClusterList.append(comp)
                    comp.pannelConnection = True
                    routeList.append([centerPoint, destination_point])
            
            # Assign cluster names to Tier 2 components
            if len(caegory2ClusterList) > 0:
                for clusterId, clusterComp in enumerate(caegory2ClusterList):
                    clusterComp.clusterName = str(routingIds) + "_" + str(clusterId)
                
                # Assign Tier 3 components to nearest Tier 2 cluster
                for comp3 in routeComponentList[routingIds]:
                    if comp3.category == 3:
                        lowestDistance = 999999999
                        for comp2 in caegory2ClusterList:
                            distance = self.distanceCalculate(comp3.centerPoint, comp2.centerPoint)
                            if distance < lowestDistance:
                                lowestDistance = distance
                                comp3.clusterName = comp2.clusterName
            
            # GRAPH-BASED TIER 3 ROUTING
            for clusterId, clusterComp in enumerate(caegory2ClusterList):
                clusterLabel = str(routingIds) + "_" + str(clusterId)
                image = self.drawTransparentTextBox(image, clusterLabel, 
                                                   (clusterComp.centerPoint[0]-50, clusterComp.centerPoint[1]-50),
                                                   bgColor=(0, 255, 255), textColor=(0, 0, 139), opacity=0.5)
                
                # Get Tier 3 components for this cluster
                tier3ComponentsInCluster = []
                for comp in routeComponentList[routingIds]:
                    if comp.category == 3 and comp.clusterName == clusterComp.clusterName:
                        tier3ComponentsInCluster.append(comp)
                
                if len(tier3ComponentsInCluster) > 0:
                    # BUILD FULLY CONNECTED GRAPH
                    G, tier2NodeId = self.buildGraphForCluster(clusterComp, tier3ComponentsInCluster)
                    
                    # Get Tier 3 node IDs
                    tier3NodeIds = [node for node in G.nodes() if G.nodes[node]['tier'] == 3]
                    
                    # FIND OPTIMAL PATH using DFS
                    optimalPath = self.dfsMinimalPath(G, tier2NodeId, tier3NodeIds)
                    
                    # VISUALIZE GRAPH - Pass routingIds and clusterId for proper naming
                    self.visualizeGraph(G, optimalPath, clusterLabel, routingIds, clusterId)
                    
                    # ROUTE THE LINEAR PATH (No branching)
                    previousComponent = clusterComp
                    
                    for idx, nodeName in enumerate(optimalPath[1:], start=1):  # Skip Tier 2
                        currentComp = G.nodes[nodeName]['component']
                        
                        # Create routing ID
                        tier3RoutingId = f"{routingIds}_{clusterId}_{idx}"
                        
                        # Add route from previous to current
                        sourcePoint = previousComponent.centerPoint
                        destinationPoint = currentComp.centerPoint
                        routeList.append([sourcePoint, destinationPoint])
                        
                        currentComp.pannelConnection = True
                        previousComponent = currentComp
                        
                        # Draw label
                        image = self.drawTransparentTextBox(image, tier3RoutingId,
                                                           (currentComp.centerPoint[0]-50, currentComp.centerPoint[1]-50),
                                                           bgColor=(0, 255, 255), textColor=(0, 0, 139), opacity=0.5)
            
            # Draw all routes using A* pathfinding with connection points
            for individualRoute in routeList:
                centerPoint, destination_point = individualRoute
                
                # Find components for this route
                sourceComp = None
                destComp = None
                
                # Search through all components in this routing group and entire componentList
                for comp in componentList:
                    # Check if this component matches the source point
                    if (comp.centerPoint == centerPoint or 
                        (hasattr(comp, 'edgePointLeft') and comp.edgePointLeft == centerPoint)):
                        sourceComp = comp
                    # Check if this component matches the destination point
                    if (comp.centerPoint == destination_point or
                        (hasattr(comp, 'edgePointLeft') and comp.edgePointLeft == destination_point)):
                        destComp = comp
                
                # If components found, try all connection point combinations
                if sourceComp and destComp:
                    sourceConnPoints = self.getAllowedConnectionPoints(sourceComp)
                    
                    # For destination, avoid reused points
                    destConnPoints = [p for p in self.getConnectionPoints(destComp) if p not in destComp.usedPoints]
                    if not destConnPoints: 
                        destConnPoints = self.getConnectionPoints(destComp)

                    bestPath = None
                    bestStartPoint = None
                    bestEndPoint = None
                    shortestDistance = float('inf')
                    
                    # Try all combinations
                    for startPoint in sourceConnPoints:
                        for endPoint in destConnPoints:
                            # USE NEW MAX BENDS ALGORITHM
                            path = map.astar_max_bends(map.pathFindingArrayStar,
                                           (startPoint[0]//SCALE, startPoint[1]//SCALE),
                                           (endPoint[0]//SCALE, endPoint[1]//SCALE),
                                           max_bends=50)
                            
                            if path is not None:
                                pathLength = len(path)
                                if pathLength < shortestDistance:
                                    shortestDistance = pathLength
                                    bestPath = path
                                    bestStartPoint = startPoint
                                    bestEndPoint = endPoint
                    
                    # Force through with center points if all connection points fail
                    if bestPath is None:
                        # Fallback for strict edge rule... (disabled)
                        pass

                    # Update used points used in successful route
                    if bestStartPoint is not None:
                        if bestStartPoint not in sourceComp.usedPoints:
                            sourceComp.usedPoints.append(bestStartPoint)
                    if bestEndPoint is not None:
                        if bestEndPoint not in destComp.usedPoints:
                            destComp.usedPoints.append(bestEndPoint)
                    
                    path = bestPath
                    actualStartPoint = bestStartPoint
                    actualEndPoint = bestEndPoint
                else:
                    # Fallback to original points if components not found
                    path = map.astar_max_bends(map.pathFindingArrayStar,
                                   (centerPoint[0]//SCALE, centerPoint[1]//SCALE),
                                   (destination_point[0]//SCALE, destination_point[1]//SCALE),
                                   max_bends=50)
                    actualStartPoint = centerPoint
                    actualEndPoint = destination_point
                
                if path is not None:
                    # Mark the path in pathFindingArrayStar (no buffer)
                    for point in path:
                        px, py = point
                        if 0 <= px < len(map.pathFindingArrayStar[0]) and 0 <= py < len(map.pathFindingArrayStar):
                            map.pathFindingArrayStar[py][px] = 0
                    
                    branch_distance = len(path) * SCALE
                    completeDistance += branch_distance
                    totalBranchDistance += branch_distance
                    
                    dataWriter += f"RouteId: {routingIds} - Branch Distance: {branch_distance} - Total: {totalBranchDistance}\n"
                    
                    segment_bends = 0
                    if len(path) > 2:
                        for i in range(2, len(path)):
                            dx1 = path[i-1][0] - path[i-2][0]
                            dy1 = path[i-1][1] - path[i-2][1]
                            dx2 = path[i][0] - path[i-1][0]
                            dy2 = path[i][1] - path[i-1][1]
                            if (dx1, dy1) != (dx2, dy2):
                                segment_bends += 1
                    total_bends += segment_bends
                    total_segments += 1
                    dataWriter += f"  Bends: {segment_bends}\n"
                    
                    
                    
                    map.image = image.copy()
                    
                    # DRAW EXACT PATH SEGMENTS (No diagonal smoothing)
                    # Convert grid coords back to pixel coords
                    pixel_path = [(p[0]*SCALE, p[1]*SCALE) for p in path]
                    
                    # Draw lines between consecutive points.
                    # Thickness 6 (was 3) for better visibility on dense diagrams.
                    for i in range(len(pixel_path) - 1):
                        cv2.line(image, pixel_path[i], pixel_path[i+1], 
                                 (int(color[0]), int(color[1]), int(color[2])), 6)
                    
                    # NOTE: removed cv2.imwrite("output/astar.jpg", image) here.
                    # That was being called once per drawn route (~hundreds of
                    # times per run), each writing a ~95-megapixel JPEG. The
                    # final image is saved once by demo.py.
                    
                    #branch_distance = len(path) * 10
                    #completeDistance += branch_distance
                    #totalBranchDistance += branch_distance
                    
                    #dataWriter += f"RouteId: {routingIds} - Branch Distance: {branch_distance} - Total: {totalBranchDistance}\n"
                    

        
        dataWriter += f"Complete distance: {completeDistance}\n"
        
        if total_segments > 0:
            avg_bends = total_bends / total_segments
            dataWriter += f"Average bends per segment: {avg_bends:.1f}\n"
            dataWriter += f"Total bends: {total_bends}, Total segments: {total_segments}\n"
        
        with open("output/data.txt", "w", encoding='utf-8') as file:
            file.write(dataWriter)
        
        return self.grid, image, dataWriter

    def drawTransparentTextBox(self, image, text, position, bgColor=(0, 255, 255), textColor=(0, 0, 139), opacity=0.5):
        """
        Draw text with transparent background.
        bgColor: BGR format (default yellow)
        textColor: BGR format (default dark blue)
        opacity: transparency level (0-1)
        """
        font = cv2.FONT_HERSHEY_COMPLEX
        fontScale = 0.6
        thickness = 1
        
        # Get text size
        (textWidth, textHeight), baseline = cv2.getTextSize(text, font, fontScale, thickness)
        
        # Define box coordinates with padding
        padding = 5
        x, y = position
        topLeft = (x - padding, y - textHeight - padding)
        bottomRight = (x + textWidth + padding, y + baseline + padding)
        
        # Create overlay for transparency
        overlay = image.copy()
        
        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, topLeft, bottomRight, bgColor, -1)
        
        # Blend overlay with original image
        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)
        
        # Draw text
        cv2.putText(image, text, (x, y), font, fontScale, textColor, thickness)
        
        return image
    
    def getConnectionPoints(self, component):
        """
        Get 4 connection points for a component (Top, Bottom, Left, Right).
        """
        return [component.edgePointTop, component.edgePointBottom, component.edgePointLeft, component.edgePointRight]

    def getAllowedConnectionPoints(self, comp):
        """
        Get allowed connection points enforcing 'different edge' rule.
        Returns any connection point that hasn't been used yet.
        """
        all_pts = self.getConnectionPoints(comp)
        
        # If no points used yet, all are available
        if len(comp.usedPoints) == 0:
            return all_pts

        # Filter used points
        available = [p for p in all_pts if p not in comp.usedPoints]
        
        # If we have available points, return them to ensure we pick a NEW edge
        if available:
            return available
            
        # If strict adherence is required but all points used, we might return empty
        # forcing the router to fail rather than reuse an edge (which would be messy)
        return []

    def distanceCalculate(self,p1, p2):
        """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
        dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
        return dis

    def nearestNeighborSort(self, tier3Components, startingPoint):
        """
        Sort Tier 3 components in nearest-neighbor order starting from the closest Tier 3 to Tier 2.
        Returns a list of components sorted by proximity (one-by-one loop using centerPoints).
        """
        if len(tier3Components) == 0:
            return []
        
        sortedList = []
        remainingComponents = tier3Components.copy()
        currentPoint = startingPoint
        
        while len(remainingComponents) > 0:
            # Find the nearest component from remaining components
            nearestComponent = None
            nearestDistance = float('inf')
            
            for comp in remainingComponents:
                # Calculate distance from current point to this component's centerPoint
                distance = self.distanceCalculate(currentPoint, comp.centerPoint)
                
                if distance < nearestDistance:
                    nearestDistance = distance
                    nearestComponent = comp
            
            if nearestComponent is not None:
                sortedList.append(nearestComponent)
                # Update current point to the centerPoint of the nearest component (for next iteration)
                currentPoint = nearestComponent.centerPoint
                remainingComponents.remove(nearestComponent)
        
        return sortedList

    #calculate the closest pannel location
    def calculateClosestPannel(self, componentList):
        # calculate the closest pannel near the symbol

        # separate the pannels and symbols
        pannels = []
        symbols = []
        for comp in componentList:
            # print(comp.name, comp.location, comp.routeId)
            if comp.name == "PANELBOARD":
                pannels.append(comp)
            else:
                symbols.append(comp)




        # calculate the distance between the symbol and the nearest pannel
        #routing the symbols in category 2
        for symbol in symbols:
            if symbol.category == 3:
                continue
            minDistance = 999999999
            minPannel = None
            for pannel in pannels:
                distance = self.distanceCalculate(
                    symbol.centerPoint, pannel.centerPoint
                )

                if distance < minDistance and self.map.verifyObstacle(symbol.centerPoint, pannel.centerPoint):
                    minDistance = distance
                    minPannel = pannel
            if minPannel != None:
                symbol.routeId = minPannel.routeId
                symbol.distanceFromPannelBoardCenter = minDistance
        
        #check if the pannels have the category 2 symbols
        for pannel in pannels:
            for symbol in symbols:
                if symbol.routeId == pannel.routeId:
                    pannel.presenceOFCategory2 = True
                    break    
            
        
        #routing the symbols in category 3
        for symbol in symbols:
            if symbol.category == 3:
                minDistance = 999999999
                minPannel = None
                for pannel in pannels:
                    if pannel.presenceOFCategory2 == False:
                        continue
                    distance = self.distanceCalculate(
                        symbol.centerPoint, pannel.centerPoint
                    )

                    if distance < minDistance and self.map.verifyObstacle(symbol.centerPoint, pannel.centerPoint):
                        minDistance = distance
                        minPannel = pannel
                if minPannel != None:
                    symbol.routeId = minPannel.routeId
                    symbol.distanceFromPannelBoardCenter = minDistance

        #chec if all the symbols has routeId
        errorCounter = 0
        for symbol in symbols:
            if symbol.routeId == -1:
                # print("Symbol not routed")
                # raise Exception("Symbol not routed")
                errorCounter += 1
        print("No of symbols not routed", errorCounter)


        # merge the pannels and symbols
        # print(pannels + symbols)
        return pannels + symbols
                





    def multilinePlotter(self,image, points,color=(80,120,90)):
        # print(i, points[i])
        x1, y1 = points[0]
        x2, y2 = points[1]

        # line distance calculator for the bend line
        branch_distance = self.distanceCalculate(
            [x1, y1], [x2, y1]
        ) + self.distanceCalculate([x2, y1], [x2, y2])

        # Draw horizontal and vertical lines to create a 90-degree bend

        cv2.line(image, (x1, y1), (x2, y1), color, 3)
        cv2.line(image, (x2, y1), (x2, y2), color, 3)

        return image,branch_distance     

    def removePointsOutsideTheObstacle(self, points, Obmap):
        # remove the points which are outside the obstacle
        #convert the image from BGR to GRAY
        Obmap = cv2.cvtColor(Obmap, cv2.COLOR_BGR2GRAY)
        newPoints = []
        for point in points:
            if (
                Obmap[
                    point.centerPoint[1] : point.centerPoint[1] + 1,
                    point.centerPoint[0] : point.centerPoint[0] + 1,
                ][0][0]
                == 255
            ):
                newPoints.append(point)
        return newPoints 

class Node:
    def __init__(self, x, y,parent=None, position=None):
        self.x = x
        self.y = y
        self.neighbours = []
        self.visited = False
        self.parent = None
        self.distance = 0
        self.cost = 0
        self.heuristic = 0

        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def add_neighbour(self, node):
        self.neighbours.append(node)

    def __str__(self):
        return f'Node {self.position}'
    
    def __eq__(self, other):
        # print("Comparing nodes", self.position, other.position)
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)
    
class Map:
    def __init__(self,width,height) -> None:
        self.height = height
        self.width = width
        self.channels = 3
        self.image = None
        self.create_map(height, width, 3)
        self.nodes = []
        self.pathFindingArray = []
        self.pathFindingArrayStar = []
        self.obstacleMap = None
        # self.create_route_map(height, width)

    def setObstacleMap(self,obMap):
        # obMap = cv2.cvtColor(obMap, cv2.COLOR_BGR2GRAY)
        self.obstacleMap = obMap
        return True

    def create_route_map(self, height, width):   
        pathFindingArray = []
        presentationImage = self.obstacleMap.copy()
        # print("Size of the obstacle map", self.obstacleMap.shape)
        # exit()

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        for j in range(int(height//(SCALE))):
            pathFindingArray.append([])
            for i in range(int(width//(SCALE))):
                # check for white and assign zeros
                # print("Checking for obstacle", i*SCALE, j*SCALE, self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].sum())
                # print("shape of obstacle map", self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].shape)
                # print("Checking for obstacle", i, j, "SCALE of width and height", width//(SCALE*SCALE), height//(SCALE*SCALE))
                
                #draw a rectangle
                cv2.rectangle(presentationImage, (i*SCALE, j*SCALE), (i*SCALE+SCALE, j*SCALE+SCALE), (0, 255, 0), 1)
                # print("Checking for obstacle", j*SCALE, i*SCALE, self.obstacleMap[j*SCALE:j*SCALE+SCALE,i*SCALE:i*SCALE+SCALE].sum())
                if self.obstacleMap[j*SCALE:(j*SCALE)+SCALE,i*SCALE:(i*SCALE)+SCALE].sum() > 0:
                    pathFindingArray[j].append(0) 
                    # print("Adding 0", i, j)
                    cv2.putText(presentationImage, "0", ((i*SCALE)+5, (j*SCALE)+5), cv2.FONT_HERSHEY_COMPLEX, 0.30, (0, 255, 0), 1)
                else:
                    # print("Checking for obstacle", i*SCALE, j*SCALE, self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].sum())
                    # print("shape of obstacle map", self.obstacleMap[i*SCALE:i*SCALE+SCALE,j*SCALE:j*SCALE+SCALE].shape)
                    # print("Adding 1", i, j) 
                    cv2.putText(presentationImage, "1", ((i*SCALE)+5, (j*SCALE)+5), cv2.FONT_HERSHEY_COMPLEX, 0.30, (0, 255, 0), 1)
                    # exit()
                    pathFindingArray[j].append(1)

        self.pathFindingArray = pathFindingArray
        # self.pathFindingArrayStar.extend([[0]*2000]*2000)
        # Deep copy the array structure to ensure it matches the map dimensions
        self.pathFindingArrayStar = [row[:] for row in self.pathFindingArray]
        
        #save the array in a text file
        with open("output/pathFindingArray.txt", "w", encoding='utf-8') as file:
            for i in pathFindingArray:
                file.write(str(i) + "\n")

        with open('output/pathFindingArray.pkl', 'wb') as f:
            pickle.dump(pathFindingArray, f)
        return True
    
    def getRouteMap(self):
        return np.asarray(self.pathFindingArray)
    
    def verifyObstacle(self, start, end):
        routingArray = self.getRouteMap()
        print("shapre of routing array", routingArray.shape)


        #check for the obstacle in the path with multiline plotter algorithm    
        print("Start and End", start, end)
        # print(i, points[i])
        x1, y1 = start
        x2, y2 = end

        # line distance calculator for the bend line
        # branch_distance = self.distanceCalculate(

        if x1 > x2:
            x1t, x2t = x2, x1
        else:
            x1t, x2t = x1, x2
        

        # print("Pathfinding Array", routingArray[y1//SCALE:y1//SCALE+1,x1t//SCALE:x2t//SCALE+1])
        loss = routingArray[y1//SCALE:y1//SCALE+1,x1t//SCALE:x2t//SCALE+1].sum()

        # cv2.imshow("image", testerImage)

        if y1 > y2:
            y1t,y2t = y2,y1
        else:
            y1t,y2t = y1,y2
        
        
        # cv2.line(testerImage, (x2, y1), (x2, y2), color, 2)
        #add the pathfinding array from (x2, y1) to (x2, y2)
        # print("Pathfinding Array", routingArray[y1t//SCALE:y2t//SCALE+1,x2//SCALE:x2//SCALE+1])
        loss += routingArray[y1t//SCALE:y2t//SCALE+1,x2//SCALE:x2//SCALE+1].sum()

        # cv2.imshow("image", testerImage)
        # cv2.waitKey(0)

        print("Loss: ", loss)
        if loss > 0:
            print("Obstacle detected and failed to route")
            return False
        else:
            return True

    def create_map(self, height, width, channels):
        image = np.zeros((height, width, channels), dtype=np.uint8)
        self.image = image
        self.height = height
        self.width = width
        self.channels = channels
        return image

    def create_nodes(self, image):
        print("Creating nodes")
        print("image shape", image.shape[0] // SCALE, image.shape[1] // SCALE)
        nodes = []
        for i in range((image.shape[1] // SCALE)+1):
            for j in range((image.shape[0] // SCALE)+1):
                node = Node(x = i * SCALE, y = j * SCALE, parent=None, position=(i * SCALE, j * SCALE))
                nodes.append(node)
        self.nodes = nodes
        return nodes



    def connectInputNodes(self,nodes,color=(0, 255, 0)):
        #draw a line connecting all the nodes in the list
        for i in range(len(nodes)-1):
            # print("Connecting nodes", nodes[i].position, nodes[i+1].position)
            cv2.line(self.image, (nodes[i].x, nodes[i].y), (nodes[i+1].x, nodes[i+1].y), color, 2)
        return self.image
    
    def nodeLocator(self, x, y):
        x = x * SCALE
        y = y * SCALE
        for node in self.nodes:
            if x - node.x < SCALE and y - node.y < SCALE:
                return node
        return None

    def astar(self, maze, start, end):
        """
        Optimized A* implementation using heapq and direct coordinate tuples.
        """
        # Ensure start and end are tuples
        start = tuple(start)
        end = tuple(end)
        
        # Dimensions: maze is [row][col] -> [y][x]
        rows = len(maze)
        cols = len(maze[0])
        
        # Priority queue: (f_score, g_score, current_pos, parent_pos)
        open_list = [(0, 0, start, None)]
        came_from = {}
        g_scores = {start: 0}
        
        path_found = False
        
        while open_list:
            f, g, current, parent = heapq.heappop(open_list)
            
            if current in g_scores and g > g_scores[current]:
                continue
            
            if current not in came_from:
                came_from[current] = parent
            
            if current == end:
                path_found = True
                break
            
            # Neighbors
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                # Check bounds: y is row, x is col
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                
                # Check walkable: maze[y][x] should be 0 (Walkable)
                if maze[ny][nx] != 0:
                    continue
                
                new_g = g + 1
                if neighbor not in g_scores or new_g < g_scores[neighbor]:
                    g_scores[neighbor] = new_g
                    h = ((neighbor[0] - end[0]) ** 2) + ((neighbor[1] - end[1]) ** 2)
                    f = new_g + h
                    heapq.heappush(open_list, (f, new_g, neighbor, current))
        
        if path_found:
            path = []
            curr = end
            while curr is not None:
                path.append(curr)
                curr = came_from.get(curr)
            return path[::-1]
        
        return None

    def astar_max_bends(self, maze, start, end, max_bends=100):
        """
        Performance Optimized A* search (Weighted Turn Penalty).
        - Moves strictly orthogonally (No Diagonals).
        - Penalizes turns to prioritize straight paths.
        - State space is small: (x, y, last_dir), making it very fast.
        """
        start = tuple(start)
        end = tuple(end)
        
        rows = len(maze)
        cols = len(maze[0])
        
        # Heuristic: Manhattan Distance
        def h_score(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority Queue: (f_score, g_score, bends, current_pos, (dx, dy))
        # Initial: cost 0, 0 bends, start pos, direction None
        open_list = [(h_score(start, end), 0, 0, start, None)]
        
        # Visited state: (x, y, dx, dy) -> min_g_score
        # Crucial Optimization: We DO NOT include 'bends' in the visited key effectively.
        # We only care about arriving at (x,y) with direction (dx,dy) with the lowest COST.
        # Since turns add cost, the lowest cost implies standard bend minimization.
        visited = {}
        
        came_from = {}
        
        TURN_PENALTY = 5  # Cost of making a turn (vs 1 for moving straight)
        
        best_path_end = None
        
        while open_list:
            f, g, bends, current, last_dir = heapq.heappop(open_list)
            
            # If we reached the goal
            if current == end:
                best_path_end = (current, last_dir)
                break
            
            # State signature
            state = (current, last_dir)
            if state in visited and visited[state] <= g:
                continue
            visited[state] = g
            
            # Neighbors: Strictly 4 directions
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Bounds check
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                
                # Walkable check
                if maze[ny][nx] != 0:
                    continue
                
                # Calculate cost and bends
                move_cost = 1
                new_bends = bends
                
                if last_dir is not None and (dx, dy) != last_dir:
                    move_cost += TURN_PENALTY
                    new_bends += 1
                
                # Hard limit check (failsafe)
                if new_bends > max_bends:
                    continue
                
                new_g = g + move_cost
                new_f = new_g + h_score((nx, ny), end)
                
                # Add to queue
                new_state = ((nx, ny), (dx, dy))
                
                if new_state not in visited or visited[new_state] > new_g:
                    heapq.heappush(open_list, (new_f, new_g, new_bends, (nx, ny), (dx, dy)))
                    came_from[new_state] = (current, last_dir)

        if best_path_end:
            # Reconstruct
            path = []
            curr_pos, curr_dir = best_path_end
            
            while curr_pos:
                path.append(curr_pos)
                parent_info = came_from.get((curr_pos, curr_dir))
                if parent_info:
                    curr_pos, curr_dir = parent_info
                else:
                    curr_pos = None
                
            return path[::-1]
            
        return None


import time

class timeit:
    def __init__(self, label="Block"):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"{self.label} took {self.end - self.start:.6f} seconds")




if __name__ == "__main__":

    



    PATH = os.getcwd()+'/input/'

    #output folder creation
    os.makedirs(os.getcwd()+'/output',exist_ok=True)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)


    for file in os.listdir(PATH):
        if file.endswith(".png"):
            start_time = time.time()
            image = cv2.imread(PATH+'/'+file)
            image  = cv2.resize(image,(WIDTH,HEIGHT))
            # WIDTH = image.shape[1]
            # HEIGHT = image.shape[0]

    
            

       





            

            drawSegmentationMap(PATH + "/" + file)

            ObstaclemapImage = cv2.imread(
            os.getcwd() + "/output/" + file[:-4] + "_obstacle.png"
        )
            

            #create a router instance
            router = Router(Obmap=ObstaclemapImage)
            #save the router instance as pickle
            with open('output/router.pkl', 'wb') as f:
                pickle.dump(router, f)
            #load the router instance from pickle
            # with open('output/router.pkl', 'rb') as f:
            #     router = pickle.load(f)
        


            #creating obstacle map

            #open the file with class names
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rule_based_class_mapping.json'), encoding='utf-8') as f:
                lines = f.read()
                id_to_name = {}
                #conver the text as json
                id_to_name = json.loads(lines)
                #get the class names)

            #components list
            components = []
            routeId = 0
            #read the yolo coordinates
            with open(PATH+'/'+file[:-4]+'.txt', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    # print(line)
                    #removing the unwanted classes as per the rules
                    if line[0] not in id_to_name:
                        continue
                    classId = int(line[0])
                    x = float(line[1])
                    y = float(line[2])
                    w = float(line[3])
                    h = float(line[4])
                    # print(classId)
                    # print(x,y,w,h)
                    l = int((x - w / 2) * WIDTH)
                    r = int((x + w / 2) * WIDTH)
                    t = int((y - h / 2) * HEIGHT)
                    b = int((y + h / 2) * HEIGHT)
                    print("class ",id_to_name[str(classId)]["name"]," vealues of r",r,"l",l,"t",t,"b",b)
                    
                    if l < 0:
                        l = 0
                    if r > WIDTH - 1:
                        r = WIDTH - 1
                    if t < 0:
                        t = 0
                    if b > HEIGHT - 1:
                        b = HEIGHT - 1
                    #plot the rectangle with condition
                    if classId == 14:
                        

                        cv2.circle(image, (int((l + r) / 2), int((t + b) / 2)), 5, (0, 255, 0), -1)
                        #create a component
                        conponentTemp = Component(id_to_name[str(classId)]["name"], (l, t, r, b),routeId=routeId,category=id_to_name[str(classId)]["category"])
                        routeId += 1
                        #plot the edge points
                        cv2.circle(image, conponentTemp.edgePointTop, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointBottom, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointLeft, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointRight, 5, (201, 138, 99), -1)
                        
                        components.append(conponentTemp)

                    else:

                        cv2.circle(image, (int((l + r) / 2), int((t + b) / 2)), 5, (0, 0, 255), -1)
                        #create a component
                        conponentTemp = Component(id_to_name[str(classId)]["name"], (l, t, r, b),category=id_to_name[str(classId)]["category"])

                        #plot the edge points
                        cv2.circle(image, conponentTemp.edgePointTop, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointBottom, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointLeft, 5, (201, 138, 99), -1)
                        cv2.circle(image, conponentTemp.edgePointRight, 5, (201, 138, 99), -1)
                        
                        components.append(conponentTemp)




            distanceOptimisedComponentList = router.calculateClosestPannel(components)
            #save the distance optimised component list as pick and load it back
            with open('output/distanceOptimisedComponentList.pkl', 'wb') as f:
                pickle.dump(distanceOptimisedComponentList, f)
            
            # with open('output/distanceOptimisedComponentList.pkl', 'rb') as f:
            #     distanceOptimisedComponentList = pickle.load(f)
            

            # remove the points which are outside the obstacle
            distanceOptimisedComponentList = router.removePointsOutsideTheObstacle(
            distanceOptimisedComponentList, ObstaclemapImage
        )

        
            with timeit("Distance Optimised Component List printing"):
                for component in distanceOptimisedComponentList:
                    cv2.putText(image, str(component.routeId), component.centerPoint, cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 1)
                    # cv2.putText(image, str(component.centerPoint), component.centerPoint, cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 1)
                    #add the class name
                    # cv2.putText(image, str(component.name), component.centerPoint, cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 1)
                    #add bounding box based on the catorgory with different colours
                    if component.category == 1:
                        image = draw_filled_rectangle_with_opacity(image, component.location[:2], component.location[2:], (0,255,0), 0.5)
                    elif component.category == 2:
                        image = draw_filled_rectangle_with_opacity(image, component.location[:2], component.location[2:], (255,0,0), 0.5)
                    else:
                        image = draw_filled_rectangle_with_opacity(image, component.location[:2], component.location[2:], (67,135,226), 0.5)
        
            cv2.imwrite("output/astar.jpg", image)
          

            with timeit("Routing"):
                #grid routing
                grid,image,datawrite = router.routeTheGridAstar(distanceOptimisedComponentList,image,Obmap=ObstaclemapImage)
                end_time =time.time()
                processing_time = end_time - start_time
                print(f"Processing time for {file}: {processing_time} seconds")
                datawrite += f"Processing time for {file}: {processing_time} seconds\n"

            # create a named window 
            # cv2.namedWindow('image', cv2.WINDOW_NORMAL)


            #save the image in the output folder
            cv2.imwrite(os.getcwd()+'/output/'+file,image)

            #save the datawriter in the output folder
            with open(os.getcwd()+'/output/'+file[:-4]+'.txt','w', encoding='utf-8') as f:
                f.write(datawrite)
