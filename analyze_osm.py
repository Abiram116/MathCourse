import osmium
import sys
from collections import Counter

# This is a simple handler that collects place names from the OSM file
class PlaceHandler(osmium.SimpleHandler):
    def __init__(self):
        super(PlaceHandler, self).__init__()
        self.places = []
        self.cities = []
        self.regions = []
        self.admin_levels = Counter()
        
    def node(self, n):
        # Process place tags on nodes
        if 'place' in n.tags:
            place_type = n.tags.get('place')
            place_name = n.tags.get('name')
            if place_name:
                self.places.append((place_type, place_name))
                
                # Collect cities
                if place_type in ['city', 'town']:
                    self.cities.append(place_name)
                
    def way(self, w):
        # Some administrative boundaries can be ways
        if 'admin_level' in w.tags:
            admin_level = w.tags.get('admin_level')
            self.admin_levels[admin_level] += 1
    
    def relation(self, r):
        # Administrative boundaries are typically relations
        if 'admin_level' in r.tags and 'name' in r.tags:
            admin_level = r.tags.get('admin_level')
            name = r.tags.get('name')
            self.admin_levels[admin_level] += 1
            
            # Collect regions (states, districts)
            if admin_level in ['4', '5', '6']:
                self.regions.append(name)

# Main function
def main(osm_file):
    handler = PlaceHandler()
    
    print(f"Analyzing OSM file: {osm_file}")
    print("This may take a while for large files...")
    
    # Process the file
    handler.apply_file(osm_file)
    
    # Display results
    print("\n=== SUMMARY ===")
    print(f"Found {len(handler.places)} named places")
    print(f"Found {len(handler.cities)} cities/towns")
    print(f"Found {len(handler.regions)} administrative regions")
    
    print("\n=== MAJOR CITIES ===")
    for city in sorted(set(handler.cities))[:50]:  # Show top 50 cities
        print(f"- {city}")
    
    print("\n=== REGIONS (States/Districts) ===")
    for region in sorted(set(handler.regions))[:50]:  # Show top 50 regions
        print(f"- {region}")
    
    print("\nNOTE: This is a sample of places. The OSM file may contain many more locations.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_osm.py <osm_file>")
        sys.exit(1)
    
    main(sys.argv[1])
