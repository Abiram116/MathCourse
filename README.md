# MathCourse - Route Optimization and Network Analysis

A Django-based web application that provides route optimization and network analysis capabilities using OpenStreetMap data. The application implements Dijkstra's algorithm to find shortest and fastest paths between locations in Hyderabad, India.

## Features

- Interactive map interface for selecting start and end points
- Real-time route calculation using Dijkstra's algorithm
- Support for both shortest distance and fastest route options
- Integration with OpenStreetMap data
- Efficient caching of network data
- Responsive web interface

## Technical Stack

- **Backend**: Django 4.2.19
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: 
  - NetworkX for graph operations
  - OSMnx for OpenStreetMap data handling
  - Shapely for geometric operations
- **Deployment**: 
  - WhiteNoise for static file serving
  - Gunicorn as WSGI server
  - Render.com compatible

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- OpenStreetMap data file for any Region of your choice

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MathCourse
```

2. Create and activate a virtual environment:
```bash
python -m venv venv_name
source venv_name/bin/activate  # On Windows: venv_name\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following:
```
DEBUG=True
SECRET_KEY=your-secret-key
```

5. Download OpenStreetMap Data:
   - Download the Hyderabad region OSM data file
   - Place it in the `cache` directory with the name `Southern Zone Latest.osm.pbf`
   - This file is required for the Dijkstra algorithm to work in local development

## Running the Application

1. Apply database migrations:
```bash
python manage.py migrate
```

2. Start the development server:
```bash
python manage.py runserver
```

3. Access the application at `http://127.0.0.1:8000/`

## Project Structure

```
MathCourse/
├── lp_app/                 # Main application
│   ├── views/             # View modules
│   ├── templates/         # HTML templates
│   └── static/            # Static files
├── MathCourse/            # Project settings
├── staticfiles/           # Collected static files
├── cache/                 # Cached OSM data
├── requirements.txt       # Project dependencies
└── manage.py             # Django management script
```

## Usage

1. Open the application in your web browser
2. Use the map interface to select start and end points
3. Choose between shortest distance or fastest route
4. View the calculated route on the map
5. See distance and time estimates for the selected route

### Important Note about Dijkstra Page Access
The Dijkstra page exists in both local and deployed versions, but there is no direct navigation button to access it. This is because the OpenStreetMap (OSM) data file is too large to be handled efficiently in the Render.com free tier deployment. While you can manually access the page by changing the web address, it will not function properly in the deployed version due to these resource limitations. The page is fully functional in the local development environment, but requires the OSM data file to be present in the `cache` directory.

## Deployment

The application is configured for deployment on Render.com. The `Procfile` and `requirements.txt` are already set up for this purpose.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the GNU General Public License (GPL). See the [LICENSE](LICENSE) file for the full license text. The GPL ensures that this software remains free and open source, allowing users to:

- Use the software for any purpose
- Study how the software works and modify it
- Distribute copies of the original software
- Distribute modified versions of the software

## Acknowledgments

- OpenStreetMap for providing the base map data
- OSMnx and NetworkX communities for their excellent libraries
- Django community for the web framework 