import osmium
import sys
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import os


class PlaceHandler(osmium.SimpleHandler):
    def __init__(self):
        super(PlaceHandler, self).__init__()
        # Hierarchical storage
        self.states = defaultdict(lambda: {
            'districts': defaultdict(lambda: {
                'mandals': defaultdict(lambda: {
                    'cities': set(),
                    'towns': set(),
                    'villages': set(),
                    'hamlets': set(),
                    'other': set()
                })
            })
        })
        
        # Direct lists for summary
        self.all_states = set()
        self.all_districts = set()
        self.all_mandals = set()
        self.all_cities = set()
        self.all_towns = set()
        self.all_villages = set()
        
        # Mapping of admin_level to type - more accurate for India
        self.admin_level_types = {
            '2': 'country',
            '4': 'state',      # States/Union Territories in India
            '5': 'division',   # Divisions/Regions within states
            '6': 'district',   # Districts
            '7': 'sub_district', # Tehsil/Taluka/Mandal
            '8': 'block',      # Block/Municipality/Town
            '9': 'village',    # Village/Ward
            '10': 'suburb'     # Suburb/Neighborhood
        }
        
        # List of actual South Indian states for filtering
        self.south_indian_states = {
            'Andhra Pradesh', 'Telangana', 'Karnataka', 'Kerala', 'Tamil Nadu', 
            'Puducherry', 'Lakshadweep', 'Andaman and Nicobar Islands'
        }
        
        # Handle places not clearly mapped to states/districts
        self.unclassified_places = defaultdict(set)
        
        # State mappings (assuming southern India states)
        self.states_mapping = {
            'Andhra Pradesh': 'Andhra Pradesh',
            'Telangana': 'Telangana',
            'Karnataka': 'Karnataka',
            'Kerala': 'Kerala',
            'Tamil Nadu': 'Tamil Nadu',
            'AP': 'Andhra Pradesh',
            'KA': 'Karnataka',
            'KL': 'Kerala',
            'TN': 'Tamil Nadu',
            'TS': 'Telangana'
        }
        
    def node(self, n):
        # Process place tags on nodes
        if 'place' in n.tags:
            place_type = n.tags.get('place')
            place_name = n.tags.get('name')
            if not place_name:
                return
            
            # Try to get state and district
            state = None
            district = None
            
            # Check for admin tags
            addr_state = n.tags.get('addr:state')
            if addr_state and addr_state in self.states_mapping:
                state = self.states_mapping[addr_state]
            
            addr_district = n.tags.get('addr:district')
            if addr_district:
                district = addr_district
            
            # If we couldn't determine state/district, put in unclassified
            if not state:
                self.unclassified_places[place_type].add(place_name)
                return
                
            # Add to appropriate collection
            if place_type == 'city':
                self.all_cities.add(place_name)
                if district:
                    self.states[state]['districts'][district]['mandals']['*']['cities'].add(place_name)
                else:
                    self.states[state]['districts']['*']['mandals']['*']['cities'].add(place_name)
            
            elif place_type == 'town':
                self.all_towns.add(place_name)
                if district:
                    self.states[state]['districts'][district]['mandals']['*']['towns'].add(place_name)
                else:
                    self.states[state]['districts']['*']['mandals']['*']['towns'].add(place_name)
            
            elif place_type == 'village':
                self.all_villages.add(place_name)
                if district:
                    self.states[state]['districts'][district]['mandals']['*']['villages'].add(place_name)
                else:
                    self.states[state]['districts']['*']['mandals']['*']['villages'].add(place_name)
            
            elif place_type == 'hamlet':
                if district:
                    self.states[state]['districts'][district]['mandals']['*']['hamlets'].add(place_name)
                else:
                    self.states[state]['districts']['*']['mandals']['*']['hamlets'].add(place_name)
            
            else:
                if district:
                    self.states[state]['districts'][district]['mandals']['*']['other'].add(place_name)
                else:
                    self.states[state]['districts']['*']['mandals']['*']['other'].add(place_name)
    
    def relation(self, r):
        # Administrative boundaries are typically relations
        if 'admin_level' in r.tags and 'name' in r.tags:
            admin_level = r.tags.get('admin_level')
            name = r.tags.get('name')
            
            if admin_level == '4':  # Only admin_level 4 is a state in India
                # Check if this is actually a South Indian state
                if name in self.south_indian_states:
                    self.all_states.add(name)
                    
                    # Create state entry if it doesn't exist
                    if name not in self.states and name not in self.states_mapping.values():
                        self.states_mapping[name] = name
                        self.states[name] = {
                            'districts': defaultdict(lambda: {
                                'mandals': defaultdict(lambda: {
                                    'cities': set(),
                                    'towns': set(),
                                    'villages': set(),
                                    'hamlets': set(),
                                    'other': set()
                                })
                            })
                        }
            
            elif admin_level in ['6']:  # Districts
                self.all_districts.add(name)
                
                # Try to identify which state this district belongs to
                state_name = r.tags.get('addr:state')
                if state_name and state_name in self.states_mapping:
                    state = self.states_mapping[state_name]
                    # Add district to state
                    if state and state in self.states:
                        self.states[state]['districts'][name] = {
                            'mandals': defaultdict(lambda: {
                                'cities': set(),
                                'towns': set(),
                                'villages': set(),
                                'hamlets': set(),
                                'other': set()
                            })
                        }
            
            elif admin_level in ['7', '8']:  # Mandals/Talukas
                self.all_mandals.add(name)
                
                # Try to identify which state/district this mandal belongs to
                state_name = r.tags.get('addr:state')
                district_name = r.tags.get('addr:district')
                
                if state_name and state_name in self.states_mapping:
                    state = self.states_mapping[state_name]
                    if state and state in self.states:
                        if district_name and district_name in self.states[state]['districts']:
                            # Add mandal to specific district
                            district = self.states[state]['districts'][district_name]
                            district['mandals'][name] = {
                                'cities': set(),
                                'towns': set(),
                                'villages': set(),
                                'hamlets': set(),
                                'other': set()
                            }
                        else:
                            # Add mandal to unspecified district
                            self.states[state]['districts']['*']['mandals'][name] = {
                                'cities': set(),
                                'towns': set(),
                                'villages': set(),
                                'hamlets': set(),
                                'other': set()
                            }


def create_pdf(handler, output_file):
    """Create a PDF report from the handler data"""
    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    heading3_style = styles['Heading3']
    normal_style = styles['Normal']
    
    # Title
    elements.append(Paragraph("Southern India OSM Data Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Introduction
    elements.append(Paragraph("This report contains information about places in South India extracted from the OSM file. The administrative hierarchy in India is: States/Union Territories → Districts → Sub-districts (Mandals/Talukas/Tehsils) → Villages/Towns.", normal_style))
    elements.append(Spacer(1, 12))
    
    # Summary stats
    elements.append(Paragraph("Summary Statistics", heading1_style))
    elements.append(Spacer(1, 6))
    
    summary_data = [
        ["Type", "Count"],
        ["States", len(handler.all_states)],
        ["Districts", len(handler.all_districts)],
        ["Mandals/Talukas", len(handler.all_mandals)],
        ["Cities", len(handler.all_cities)],
        ["Towns", len(handler.all_towns)],
        ["Villages", len(handler.all_villages)]
    ]
    
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 12))
    
    # States and their hierarchies
    elements.append(Paragraph("Hierarchical Organization", heading1_style))
    elements.append(Spacer(1, 6))
    
    # For each state
    for state_name, state_data in sorted(handler.states.items()):
        if state_name == '*':
            continue
            
        elements.append(Paragraph(f"State: {state_name}", heading1_style))
        elements.append(Spacer(1, 6))
        
        # Districts in this state
        districts = state_data['districts']
        for district_name, district_data in sorted(districts.items()):
            if district_name == '*':
                continue
                
            elements.append(Paragraph(f"District: {district_name}", heading2_style))
            elements.append(Spacer(1, 4))
            
            # Mandals in this district
            mandals = district_data['mandals']
            for mandal_name, mandal_data in sorted(mandals.items()):
                if mandal_name == '*':
                    continue
                    
                elements.append(Paragraph(f"Mandal/Taluka: {mandal_name}", heading3_style))
                
                # Cities
                if mandal_data['cities']:
                    elements.append(Paragraph(f"Cities ({len(mandal_data['cities'])}):", normal_style))
                    city_text = ", ".join(sorted(mandal_data['cities']))
                    elements.append(Paragraph(city_text, normal_style))
                    elements.append(Spacer(1, 4))
                
                # Towns
                if mandal_data['towns']:
                    elements.append(Paragraph(f"Towns ({len(mandal_data['towns'])}):", normal_style))
                    town_text = ", ".join(sorted(mandal_data['towns']))
                    elements.append(Paragraph(town_text, normal_style))
                    elements.append(Spacer(1, 4))
                
                # Villages (limit to first 50 if too many)
                if mandal_data['villages']:
                    elements.append(Paragraph(f"Villages ({len(mandal_data['villages'])}):", normal_style))
                    villages_sorted = sorted(mandal_data['villages'])
                    if len(villages_sorted) > 50:
                        village_text = ", ".join(villages_sorted[:50]) + f"... (and {len(villages_sorted)-50} more)"
                    else:
                        village_text = ", ".join(villages_sorted)
                    elements.append(Paragraph(village_text, normal_style))
                
                elements.append(Spacer(1, 8))
            
            # Unclassified places in this district (places not assigned to a specific mandal)
            unclassified = mandals.get('*', {})
            if (unclassified.get('cities') or unclassified.get('towns') or 
                unclassified.get('villages') or unclassified.get('other')):
                
                elements.append(Paragraph(f"Other places in {district_name} (not assigned to a specific mandal):", heading3_style))
                
                # Cities
                if unclassified.get('cities'):
                    elements.append(Paragraph(f"Cities ({len(unclassified['cities'])}):", normal_style))
                    city_text = ", ".join(sorted(unclassified['cities']))
                    elements.append(Paragraph(city_text, normal_style))
                    elements.append(Spacer(1, 4))
                
                # Towns
                if unclassified.get('towns'):
                    elements.append(Paragraph(f"Towns ({len(unclassified['towns'])}):", normal_style))
                    town_text = ", ".join(sorted(unclassified['towns']))
                    elements.append(Paragraph(town_text, normal_style))
                    elements.append(Spacer(1, 4))
                
                # Villages (limit to first 30 if too many)
                if unclassified.get('villages'):
                    elements.append(Paragraph(f"Villages ({len(unclassified['villages'])}):", normal_style))
                    villages_sorted = sorted(unclassified['villages'])
                    if len(villages_sorted) > 30:
                        village_text = ", ".join(villages_sorted[:30]) + f"... (and {len(villages_sorted)-30} more)"
                    else:
                        village_text = ", ".join(villages_sorted)
                    elements.append(Paragraph(village_text, normal_style))
                
                elements.append(Spacer(1, 8))
        
        # Unclassified places in this state (places not assigned to a specific district)
        unclassified_district = districts.get('*', {})
        unclassified_mandals = unclassified_district.get('mandals', {}).get('*', {})
        
        if (unclassified_mandals.get('cities') or unclassified_mandals.get('towns') or 
            unclassified_mandals.get('villages') or unclassified_mandals.get('other')):
            
            elements.append(Paragraph(f"Other places in {state_name} (not assigned to a specific district):", heading2_style))
            
            # Cities
            if unclassified_mandals.get('cities'):
                elements.append(Paragraph(f"Cities ({len(unclassified_mandals['cities'])}):", normal_style))
                city_text = ", ".join(sorted(unclassified_mandals['cities']))
                elements.append(Paragraph(city_text, normal_style))
                elements.append(Spacer(1, 4))
            
            # Towns
            if unclassified_mandals.get('towns'):
                elements.append(Paragraph(f"Towns ({len(unclassified_mandals['towns'])}):", normal_style))
                town_text = ", ".join(sorted(unclassified_mandals['towns']))
                elements.append(Paragraph(town_text, normal_style))
                elements.append(Spacer(1, 4))
            
            # Villages (limit to first 20 if too many)
            if unclassified_mandals.get('villages'):
                elements.append(Paragraph(f"Villages ({len(unclassified_mandals['villages'])}):", normal_style))
                villages_sorted = sorted(unclassified_mandals['villages'])
                if len(villages_sorted) > 20:
                    village_text = ", ".join(villages_sorted[:20]) + f"... (and {len(villages_sorted)-20} more)"
                else:
                    village_text = ", ".join(villages_sorted)
                elements.append(Paragraph(village_text, normal_style))
            
            elements.append(Spacer(1, 8))
    
    # Unclassified places (not clearly associated with any state)
    if handler.unclassified_places:
        elements.append(Paragraph("Unclassified Places (State unknown)", heading1_style))
        elements.append(Spacer(1, 6))
        
        for place_type, places in sorted(handler.unclassified_places.items()):
            if places:
                elements.append(Paragraph(f"{place_type.title()} ({len(places)}):", heading3_style))
                places_sorted = sorted(places)
                if len(places_sorted) > 100:
                    places_text = ", ".join(places_sorted[:100]) + f"... (and {len(places_sorted)-100} more)"
                else:
                    places_text = ", ".join(places_sorted)
                elements.append(Paragraph(places_text, normal_style))
                elements.append(Spacer(1, 6))
    
    # Build PDF
    doc.build(elements)


def main(osm_file):
    handler = PlaceHandler()
    
    print(f"Analyzing OSM file: {osm_file}")
    print("This may take a while for large files...")
    
    # Process the file
    handler.apply_file(osm_file)
    
    # Create PDF report
    output_file = "southern_india_osm_report.pdf"
    print(f"\nGenerating PDF report: {output_file}")
    create_pdf(handler, output_file)
    
    print(f"PDF report created successfully: {os.path.abspath(output_file)}")
    
    # Display summary info in terminal
    print("\n=== SUMMARY ===")
    print(f"States: {len(handler.all_states)}")
    print(f"Districts: {len(handler.all_districts)}")
    print(f"Mandals/Talukas: {len(handler.all_mandals)}")
    print(f"Cities: {len(handler.all_cities)}")
    print(f"Towns: {len(handler.all_towns)}")
    print(f"Villages: {len(handler.all_villages)}")


if __name__ == "__main__":
    # Use command line argument or default to the Southern Zone Latest.osm.pbf
    if len(sys.argv) > 1:
        osm_file = sys.argv[1]
    else:
        osm_file = "data/Southern Zone Latest.osm.pbf"
    
    main(osm_file)
