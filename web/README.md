# VMEvalKit Web Dashboard 🎥

A modern web interface to visualize and explore video generation results from VMEvalKit experiments.

## Features

- 📊 **Overview Dashboard**: View statistics across all models and domains
- 🤖 **Model Performance**: Detailed analysis per model
- 🧠 **Domain Analysis**: Results grouped by reasoning domain (Chess, Maze, Raven, Rotation, Sudoku)
- 📝 **Task Comparison**: Compare how different models perform on the same task
- ⚖️ **Side-by-Side Comparison**: Matrix view to compare all results
- 🎬 **Video Playback**: View generated videos directly in the browser
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

## Screenshots

The dashboard displays:
- Total inference statistics
- Success rates by model and domain
- Video grid with playback controls
- Comparison matrices
- Task-specific details

## Installation

### 1. Navigate to the web directory

```bash
cd web
```

### 2. Install dependencies

Using the main venv (recommended):
```bash
source ../venv/bin/activate
pip install -r requirements.txt
```

Or create a separate venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

### Development Mode

```bash
python app.py
```

The dashboard will be available at: **http://localhost:5000**

### Production Mode (with Gunicorn)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

## Configuration

The dashboard automatically reads from `../data/outputs/` directory. You can customize the output directory by modifying `app.py`:

```python
app.config['OUTPUT_DIR'] = Path('/custom/path/to/outputs')
```

## API Endpoints

The dashboard also provides REST API endpoints:

### Get All Results
```
GET /api/results
GET /api/results?model=luma-ray-2
GET /api/results?domain=chess
GET /api/results?task_id=maze_0001
```

### Get Statistics
```
GET /api/statistics
```

Returns JSON with model and domain statistics.

## Directory Structure

```
web/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── utils/
│   ├── __init__.py
│   └── data_loader.py         # Data scanning and loading utilities
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Dashboard overview
│   ├── model.html             # Model-specific view
│   ├── domain.html            # Domain-specific view
│   ├── task.html              # Task comparison view
│   ├── compare.html           # Comparison matrix
│   └── error.html             # Error page
└── static/                     # Static assets
    ├── css/
    │   └── style.css          # Dashboard styles
    └── js/
        └── main.js            # Interactive features
```

## Features in Detail

### Overview Dashboard (`/`)
- Total inference count and success rate
- Model performance table with success rates
- Domain statistics with task counts
- Recent results grid with video previews

### Model View (`/model/<model_name>`)
- All results for a specific model
- Performance breakdown by domain
- Video grid with all generated videos

### Domain View (`/domain/<domain_name>`)
- All results for a specific reasoning domain
- Performance breakdown by model
- Domain-specific statistics

### Task View (`/task/<task_id>`)
- Compare all model results for a single task
- Side-by-side video comparison
- Input/output image display
- Metadata and prompt information

### Comparison Matrix (`/compare`)
- Grid view of all tasks × all models
- Video playback controls
- Quick visual comparison

## Deployment Options

### Option 1: Local Network

Run on your local machine and access from other devices on the same network:

```bash
python app.py
# Access via http://<your-ip>:5000
```

### Option 2: Cloud Deployment (DigitalOcean, AWS, etc.)

1. Clone the repository on your server
2. Install dependencies
3. Run with gunicorn:

```bash
gunicorn --bind 0.0.0.0:80 --workers 4 app:app
```

### Option 3: Docker (Future Enhancement)

A Dockerfile can be added for containerized deployment.

## Troubleshooting

### Videos not loading
- Ensure the output directory path is correct
- Check that video files exist in `data/outputs/`
- Verify video files are in MP4 format

### Performance issues
- Large datasets may take time to scan on first load
- Consider adding caching for production use
- Use gunicorn with multiple workers

### Port already in use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Mobile browsers: ✅ Responsive design

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Video**: HTML5 Video API
- **Design**: Modern dark theme with gradient accents
- **Icons**: Emoji (universal support)

## Future Enhancements

Potential improvements:
- [ ] Caching for faster load times
- [ ] Advanced filtering and sorting
- [ ] Download results as CSV/JSON
- [ ] Real-time updates via WebSocket
- [ ] Video quality analysis metrics
- [ ] Export comparison reports
- [ ] User authentication
- [ ] Docker containerization

## Contributing

The dashboard is part of VMEvalKit. Contributions welcome!

## License

Same as VMEvalKit main project.

---

**Need help?** Check the main VMEvalKit documentation or open an issue on GitHub.

