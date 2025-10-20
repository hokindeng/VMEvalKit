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

- 📊 View hierarchical organization of results (Models → Domains → Tasks)
- 🤖 Browse results organized by model
- 🧠 See tasks grouped by reasoning domain
- 🎬 Watch generated videos directly in the browser
- 📷 View input images and prompts for each task
- 🔍 Access results via REST API

## Features

### Dashboard Home (`/`)

The main dashboard displays:
- **Overview Statistics**: Total inferences, models tested, domains covered
- **Hierarchical View**: Results organized as Models → Domains → Tasks
- **Collapsible Sections**: Expand/collapse each model and domain
- **Quick Navigation**: Jump buttons to quickly access any model section
- **Auto-expand**: First model section opens automatically
- **Video Playback**: Click videos to play/pause
- **Lazy Loading**: Videos load only when sections are expanded

## Architecture

### Backend (Flask)

```
web/
├── app.py              # Main Flask application
│                       # Routes: /, /api/results
│                       # Media serving: /video/<path>, /image/<path>
└── utils/
    └── data_loader.py  # Scans output folders and loads data from filesystem
```

### Frontend

```
web/
├── templates/          # Jinja2 HTML templates
│   ├── base.html      # Base layout
│   ├── index.html     # Main dashboard with hierarchical view
│   └── error.html     # Error page
└── static/
    ├── css/
    │   └── style.css  # Apple-style clean theme
    └── js/
        └── main.js    # Video controls, lazy loading, accessibility
```

## Data Source

The dashboard automatically scans the `data/outputs/pilot_experiment/` directory structure:

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
- Model name (from folder structure)
- Domain and task ID (from folder names)
- Video path (*.mp4, *.webm, *.avi, *.mov files)
- Input images (first_frame.png, final_frame.png)
- Prompt text (from prompt.txt file)
- Timestamp (from folder modification time)

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
      "run_id": "wavespeed-wan-2.2-i2v-720p_chess_0001_20251017_071644",
      "model": "wavespeed-wan-2.2-i2v-720p",
      "domain": "chess",
      "task_id": "chess_0001",
      "timestamp": "2024-10-17T07:16:44",
      "prompt": "Show the next chess move...",
      "video_path": "/path/to/video.mp4",
      "first_frame": "/path/to/input.png",
      "final_frame": "/path/to/output.png",
      "inference_dir": "/path/to/run/directory"
    }
  ]
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

### Clean Apple-Style Theme

- **Color Palette**:
  - Background: Light gray (#f5f5f7)
  - Text: Dark (#1d1d1f)
  - Accent: System blue
  - Borders: Subtle gray
  
- **Layout**: Clean hierarchical sections with collapsible panels
- **Typography**: System fonts (-apple-system, BlinkMacSystemFont)
- **Icons**: Emoji for universal support (♠️, 🌀, 🧩, 🔄, 🔢)
- **Interactions**: Click-to-expand sections, click-to-play videos

### Responsive Design

- Desktop: Full hierarchical view with side-by-side content
- Tablet: Adaptive layouts
- Mobile: Single-column responsive design

## Browser Support

- ✅ Chrome/Edge (full support)
- ✅ Firefox (full support)
- ✅ Safari (full support)
- ✅ Mobile browsers (responsive)

## Performance

- Lazy loading for videos (only load when sections expand)
- Videos initially set to `preload="none"`
- Upgraded to `preload="metadata"` when sections expand
- IntersectionObserver for enhanced loading when videos approach viewport
- Efficient directory scanning with deduplication
- LRU cache for scan results

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
2. Verify outputs exist in `data/outputs/pilot_experiment/`

## Development

### Adding New Views

1. Create route in `app.py`:
   ```python
   @app.route('/myview')
   def my_view():
       return render_template('myview.html')
   ```

2. Create template in `templates/myview.html`
3. Update the dashboard accordingly

### Styling

All styles are in `static/css/style.css` using CSS variables for easy theming.

### JavaScript

Interactive features in `static/js/main.js`:
- Video player controls (click to play/pause)
- Lazy loading with IntersectionObserver
- Accessibility features (keyboard navigation, ARIA)
- Progress bar animations
- Error handling for failed video loads
- Notification system
- Keyboard shortcuts (Ctrl/Cmd+K for search, Escape to clear)

## Future Enhancements

- [ ] Model-specific views (`/model/<model_name>`)
- [ ] Domain-specific views (`/domain/<domain_name>`)
- [ ] Task comparison views (`/task/<task_id>`)
- [ ] Side-by-side comparison matrix
- [ ] Statistics API endpoint
- [ ] Success/failure tracking
- [ ] Generation duration metrics
- [ ] Advanced filtering and search
- [ ] Export functionality

## Integration with VMEvalKit

The dashboard is a standalone app but tightly integrated:

1. **Data Flow**: Reads from VMEvalKit's structured output folders
2. **No Modification**: Doesn't modify any experiment data
3. **Real-time**: Reflects latest experiments automatically
4. **Filesystem-based**: Reads prompt.txt and finds video/image files directly

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

