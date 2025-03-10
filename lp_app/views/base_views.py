from django.shortcuts import render
from django.http import JsonResponse
import networkx as nx
import osmnx as ox
import numpy as np
from shapely.geometry import LineString
import os
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(point1, point2):
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert coordinates to radians
    lat1, lon1 = radians(point1['lat']), radians(point1['lng'])
    lat2, lon2 = radians(point2['lat']), radians(point2['lng'])
    
    # Differences in coordinates
    dlat, dlon = lat2 - lat1, lon2 - lon1
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Distance
    distance = R * c
    
    return distance

def home(request):
    return render(request, 'home.html')

def dijkstra_view(request):
    return render(request, 'dijkstra.html')

def shortest_path(request):
    try:
        start_lat = float(request.GET.get('start_lat'))
        start_lng = float(request.GET.get('start_lng'))
        end_lat = float(request.GET.get('end_lat'))
        end_lng = float(request.GET.get('end_lng'))

        print(f"Calculating route from ({start_lat}, {start_lng}) to ({end_lat}, {end_lng})")

        # Download the street network for the bounding box of our points
        BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        cache_dir = os.path.join(BASE_DIR, 'data')
        cached_graph_file = os.path.join(cache_dir, 'hyderabad_network.graphml')
        osm_file = os.path.join(cache_dir, 'Southern Zone Latest.osm.pbf')

        try:
            # Check if the cache directory exists
            if os.path.exists(cache_dir):
                # Load from JSON if available
                json_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
                if json_files:
                    for json_file in json_files:
                        # Load the graph from the JSON file
                        G = ox.load_graphml(os.path.join(cache_dir, json_file))
                        print(f"Loaded graph from JSON file: {json_file} with {len(G.nodes)} nodes and {len(G.edges)} edges")
                        break
                else:
                    print("No JSON cache files found. Proceeding to download OSM data.")
                    # Proceed to download OSM data
                    G = ox.graph_from_place("Hyderabad, Telangana, India", network_type='drive')
                    print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
            else:
                print("Cache directory not found. Downloading OSM data.")
                # Proceed to download OSM data
                G = ox.graph_from_place("Hyderabad, Telangana, India", network_type='drive')
                print(f"Downloaded graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
        except Exception as e:
            print(f"Error processing network data: {str(e)}")
            return JsonResponse({'error': f'Unable to process street network data: {str(e)}'}, status=400)

        # Find the nearest network nodes to our points
        try:
            print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

            if len(G.nodes) == 0:
                print("No nodes found in the graph. The OSM data might be incomplete for this area.")
                return JsonResponse({'error': 'No road network found in this area. Try a different location.'}, status=400)

            try:
                # First try to find exact matches
                start_node = ox.distance.nearest_nodes(G, start_lng, start_lat)
                end_node = ox.distance.nearest_nodes(G, end_lng, end_lat)
                print(f"Found nodes: Start node: {start_node}, End node: {end_node}")
            except Exception as e1:
                print(f"Error finding nearest nodes with default parameters: {str(e1)}")
                try:
                    # Try with explicit return_dist parameter
                    start_node_tuple = ox.distance.nearest_nodes(G, start_lng, start_lat, return_dist=True)
                    end_node_tuple = ox.distance.nearest_nodes(G, end_lng, end_lat, return_dist=True)
                    # Extract just the node ID if a tuple was returned
                    start_node = start_node_tuple[0] if isinstance(start_node_tuple, tuple) else start_node_tuple
                    end_node = end_node_tuple[0] if isinstance(end_node_tuple, tuple) else end_node_tuple
                    print(f"Fallback found nodes: Start node: {start_node}, End node: {end_node}")
                except Exception as e2:
                    print(f"Second attempt to find nodes failed: {str(e2)}")
                    # Last resort: try to find the nodes using a different approach
                    try:
                        nodes = list(G.nodes)
                        if nodes:
                            # Get all node positions
                            node_coords = [[G.nodes[node]['x'], G.nodes[node]['y'], node] for node in nodes]
                            # Find closest to start using manual distance calculation
                            start_node = min(node_coords, key=lambda x: (x[0]-start_lng)**2 + (x[1]-start_lat)**2)[2]
                            # Find closest to end using manual distance calculation
                            end_node = min(node_coords, key=lambda x: (x[0]-end_lng)**2 + (x[1]-end_lat)**2)[2]
                            print(f"Manual lookup found nodes: Start node: {start_node}, End node: {end_node}")
                        else:
                            raise ValueError("No nodes available in the graph")
                    except Exception as e3:
                        print(f"All node finding attempts failed: {str(e3)}")
                        return JsonResponse({'error': 'Could not find network nodes near the selected points. Try selecting points closer to roads.'}, status=400)

            if start_node is None or end_node is None:
                return JsonResponse({'error': 'Could not find network nodes near the selected points. Try selecting points closer to roads.'}, status=400)

            print(f"Using nodes: Start node: {start_node}, End node: {end_node}")

        except Exception as e:
            print(f"Error finding nearest nodes: {str(e)}")
            return JsonResponse({'error': 'Could not find network nodes near the selected points'}, status=400)

        # Run Dijkstra's algorithm with multiple paths
        try:
            if not G.has_node(start_node) or not G.has_node(end_node):
                print("Start or end node is not connected to the road network.")
                return JsonResponse({'error': 'Start or end point is not reachable. Please try different points.'}, status=400)

            shortest_route = nx.shortest_path(G, start_node, end_node, weight='length')
            fastest_route = nx.shortest_path(G, start_node, end_node, weight='travel_time')

            paths = {'shortest': shortest_route, 'fastest': fastest_route}
            path_coords = []
            distances = {'shortest': 0, 'fastest': 0}

            for path_key, route in paths.items():
                edges = list(zip(route[:-1], route[1:]))
                path_coords.append({
                    'type': path_key,
                    'coordinates': []
                })

                path_coords[-1]['coordinates'].append({
                    'lat': G.nodes[route[0]]['y'],
                    'lng': G.nodes[route[0]]['x']
                })

                for u, v in edges:
                    try:
                        data = G.get_edge_data(u, v, 0)
                        if 'geometry' in data:
                            xs, ys = data['geometry'].xy
                            for x, y in zip(xs, ys):
                                path_coords[-1]['coordinates'].append({
                                    'lat': y,
                                    'lng': x
                                })
                                distances[path_key] += haversine_distance({'lat': G.nodes[u]['y'], 'lng': G.nodes[u]['x']}, {'lat': y, 'lng': x})
                        else:
                            path_coords[-1]['coordinates'].append({
                                'lat': G.nodes[v]['y'],
                                'lng': G.nodes[v]['x']
                            })
                            distances[path_key] += haversine_distance({'lat': G.nodes[u]['y'], 'lng': G.nodes[u]['x']}, {'lat': G.nodes[v]['y'], 'lng': G.nodes[v]['x']})
                    except Exception as e:
                        print(f"Error processing edge {u}-{v}: {str(e)}")
                        continue

                path_coords[-1]['coordinates'].append({
                    'lat': G.nodes[end_node]['y'],
                    'lng': G.nodes[end_node]['x']
                })

                distances[path_key] += haversine_distance({'lat': G.nodes[route[-2]]['y'], 'lng': G.nodes[route[-2]]['x']}, {'lat': G.nodes[end_node]['y'], 'lng': G.nodes[end_node]['x']})

            print(f"Returning paths with {len(path_coords)} routes and distances: {distances}")
            return JsonResponse({'paths': path_coords, 'distances': distances})

        except nx.NetworkXNoPath:
            print("No path found between nodes")
            return JsonResponse({'error': 'No path found between the selected points'}, status=404)
        except Exception as e:
            print(f"Error calculating shortest path: {str(e)}")
            return JsonResponse({'error': f'Error calculating path: {str(e)}'}, status=400)

    except Exception as e:
        print(f"Unexpected error in shortest_path: {str(e)}")
        return JsonResponse({'error': 'An unexpected error occurred'}, status=500)