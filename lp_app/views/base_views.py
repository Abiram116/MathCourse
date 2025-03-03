from django.shortcuts import render
from django.http import JsonResponse
import networkx as nx
import osmnx as ox

def home(request):
    return render(request, 'home.html')

def dijkstra_view(request):
    return render(request, 'dijkstra.html')

def shortest_path(request):
    start_lat = float(request.GET.get('start_lat'))
    start_lng = float(request.GET.get('start_lng'))
    end_lat = float(request.GET.get('end_lat'))
    end_lng = float(request.GET.get('end_lng'))
    
    # Download the street network for the bounding box of our points
    G = ox.graph_from_bbox(
        max(start_lat, end_lat) + 0.01, 
        min(start_lat, end_lat) - 0.01,
        max(start_lng, end_lng) + 0.01, 
        min(start_lng, end_lng) - 0.01, 
        network_type='drive'
    )
    
    # Find the nearest network nodes to our points
    start_node = ox.distance.nearest_nodes(G, start_lng, start_lat)
    end_node = ox.distance.nearest_nodes(G, end_lng, end_lat)
    
    # Run Dijkstra's algorithm
    route = nx.shortest_path(G, start_node, end_node, weight='length')
    
    # Get the coordinates for the path
    path_coords = []
    for node in route:
        path_coords.append({
            'lat': G.nodes[node]['y'],
            'lng': G.nodes[node]['x']
        })
    
    return JsonResponse({'path': path_coords})