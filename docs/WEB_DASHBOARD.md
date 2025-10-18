# VMEvalKit Web Dashboard

A modern web application for visualizing and exploring video generation results from VMEvalKit experiments.

## Quick Start

```bash
cd web
./start.sh
```

Then open your browser to: **http://localhost:5000**

## Overview

The web dashboard provides an intuitive interface to:

- 📊 View aggregate statistics across all experiments
- 🤖 Analyze performance by model
- 🧠 Explore results by reasoning domain
- 📝 Compare how different models tackle the same task
- ⚖️ Side-by-side comparison matrix
- 🎬 Watch generated videos directly in the browser

## Features

### Dashboard Home (`/`)

The main dashboard displays:
- **Overview Statistics**: Total inferences, models tested, success rates
- **Model Performance Table**: Success rates, average duration, domains covered
- **Domain Cards**: Statistics for each reasoning domain (Chess, Maze, Raven, Rotation, Sudoku)
- **Recent Results**: Grid of latest generated videos

### Model View (`/model/<model_name>`)

Detailed view for a specific model showing:
- Performance breakdown by domain
- All generated videos
- Success/failure statistics
- Generation duration metrics

### Domain View (`/domain/<domain_name>`)

View all results for a reasoning domain:
- Performance breakdown by model
- All task results
- Domain-specific statistics

### Task View (`/task/<task_id>`)

Compare how different models performed on the same task:
- Side-by-side video comparison
- Input image (first frame)
- Generated video (model output)
- Target image (final frame)
- Prompt and metadata
- Generation time and status

### Comparison Matrix (`/compare`)

Grid view showing all tasks × all models:
- Interactive video grid
- Play/pause controls
- Quick visual comparison
- Duration overlays

## Architecture

### Backend (Flask)

```
web/
├── app.py              # Main Flask application
│                       # Routes: /, /model, /domain, /task, /compare
│                       # API: /api/results, /api/statistics
│                       # Media: /video, /image
└── utils/
    └── data_loader.py  # Scans output folders and loads metadata
```

### Frontend

```
web/
├── templates/          # Jinja2 HTML templates
│   ├── base.html      # Base layout with navbar
│   ├── index.html     # Dashboard overview
│   ├── model.html     # Model-specific view
│   ├── domain.html    # Domain-specific view
│   ├── task.html      # Task comparison
│   ├── compare.html   # Comparison matrix
│   └── error.html     # Error page
└── static/
    ├── css/
    │   └── style.css  # Modern dark theme with gradients
    └── js/
        └── main.js    # Interactive features
```

## Data Source

The dashboard automatically scans the `data/outputs/` directory structure:

```
data/outputs/
└── {model}/
    └── {domain}_task/
        └── {task_id}/
            └── {run_id}/
                ├── video/
                │   └── generated_video.mp4
                ├── question/
                │   ├── first_frame.png
                │   ├── final_frame.png
                │   ├── prompt.txt
                │   └── question_metadata.json
                └── metadata.json
```

Each inference folder is parsed to extract:
- Model name and parameters
- Success/failure status
- Generation duration
- Video path
- Input/output images
- Prompt text
- Task metadata

## API Endpoints

### GET `/api/results`

Get all inference results as JSON.

**Query Parameters:**
- `model` - Filter by model name
- `domain` - Filter by domain name
- `task_id` - Filter by task ID

**Example:**
```bash
curl http://localhost:5000/api/results?model=luma-ray-2&domain=chess
```

**Response:**
```json
{
  "total": 15,
  "results": [
    {
      "run_id": "luma-ray-2_chess_0001_...",
      "model": "luma-ray-2",
      "domain": "chess",
      "task_id": "chess_0001",
      "success": true,
      "duration_seconds": 42.3,
      "video_path": "...",
      "timestamp": "2025-10-18T..."
    }
  ]
}
```

### GET `/api/statistics`

Get aggregate statistics.

**Response:**
```json
{
  "models": {
    "luma-ray-2": {
      "total": 75,
      "success": 68,
      "failed": 7,
      "success_rate": 90.7,
      "avg_duration": 38.5,
      "domains": ["chess", "maze", "raven", "rotation", "sudoku"]
    }
  },
  "domains": {
    "chess": {
      "total": 90,
      "success": 82,
      "failed": 8,
      "success_rate": 91.1,
      "models": ["luma-ray-2", "veo-3.0-generate", ...]
    }
  },
  "total_inferences": 450
}
```

## Installation

### Option 1: Use Main venv (Recommended)

```bash
cd web
source ../venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Option 2: Separate venv

```bash
cd web
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Deployment

### Development

```bash
python app.py
# Runs on http://localhost:5000
```

### Production (Gunicorn)

```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

### Environment Variables

No environment variables required - the dashboard uses relative paths to find the output directory.

## Design

### Modern Dark Theme

- **Color Palette**:
  - Primary: Blue (#2563eb)
  - Secondary: Purple (#7c3aed)
  - Success: Green (#10b981)
  - Warning: Orange (#f59e0b)
  - Danger: Red (#ef4444)
  
- **Layout**: Responsive grid system
- **Typography**: System fonts for fast loading
- **Icons**: Emoji for universal support
- **Animations**: Smooth transitions and hover effects

### Responsive Design

- Desktop: Multi-column grids
- Tablet: Adaptive layouts
- Mobile: Single-column stacks

## Browser Support

- ✅ Chrome/Edge (full support)
- ✅ Firefox (full support)
- ✅ Safari (full support)
- ✅ Mobile browsers (responsive)

## Performance

- Lazy loading for videos
- Metadata caching
- Efficient directory scanning
- Progressive loading

## Troubleshooting

### Videos Not Loading

1. Check output directory path
2. Verify video files exist
3. Check file permissions
4. Ensure MP4 format

### Port Already in Use

Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### No Data Displayed

1. Run some inferences first:
   ```bash
   python examples/experiment_2025-10-14.py
   ```
2. Verify outputs exist in `data/outputs/`

## Development

### Adding New Views

1. Create route in `app.py`:
   ```python
   @app.route('/myview')
   def my_view():
       return render_template('myview.html')
   ```

2. Create template in `templates/myview.html`
3. Add navigation link in `templates/base.html`

### Styling

All styles are in `static/css/style.css` using CSS variables for easy theming.

### JavaScript

Interactive features in `static/js/main.js`:
- Video player enhancements
- Lazy loading
- Search/filter
- Keyboard shortcuts

## Future Enhancements

- [ ] Real-time updates via WebSocket
- [ ] Advanced filtering and search
- [ ] Export to CSV/JSON
- [ ] Video quality metrics
- [ ] User authentication
- [ ] Docker containerization
- [ ] Caching layer for performance

## Integration with VMEvalKit

The dashboard is a standalone app but tightly integrated:

1. **Data Flow**: Reads from VMEvalKit's structured output folders
2. **No Modification**: Doesn't modify any experiment data
3. **Real-time**: Reflects latest experiments automatically
4. **Metadata**: Uses VMEvalKit's metadata format

## Contributing

To contribute to the web dashboard:

1. Follow VMEvalKit's contribution guidelines
2. Test on multiple browsers
3. Ensure responsive design
4. Update documentation

## License

Same as VMEvalKit main project (Apache 2.0).

---

For more information, see the [main README](../README.md) or `web/README.md`.

