#!/usr/bin/env python3
"""
Create a simple test image that might pass Luma's moderation.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Create a more photorealistic-looking image with gradient background
width, height = 1280, 720

# Create gradient background (sky-like)
image = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(image)

# Create a gradient from light blue to darker blue
for y in range(height):
    # Gradient from light blue at top to darker blue at bottom
    r = int(135 + (185 - 135) * (1 - y / height))
    g = int(206 + (225 - 206) * (1 - y / height))
    b = int(235 + (245 - 235) * (1 - y / height))
    draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

# Add some "ground" at the bottom
ground_height = height // 4
for y in range(height - ground_height, height):
    # Green gradient for grass
    intensity = (y - (height - ground_height)) / ground_height
    r = int(34 + 20 * intensity)
    g = int(139 - 30 * intensity)
    b = int(34 + 20 * intensity)
    draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

# Add a simple path with more organic look
path_width = 80
path_color = (139, 90, 43)  # Brown path

# Create a winding path
points = []
for i in range(6):
    x = int(width * 0.2 + (width * 0.6 * i / 5))
    y = int(height * 0.5 + 100 * np.sin(i * 0.8))
    points.append((x, y))

# Draw path segments
for i in range(len(points) - 1):
    x1, y1 = points[i]
    x2, y2 = points[i + 1]
    # Draw wider path
    draw.ellipse([x1 - path_width//2, y1 - path_width//2, 
                  x1 + path_width//2, y1 + path_width//2], 
                 fill=path_color)

# Connect with lines
for i in range(len(points) - 1):
    draw.line([points[i], points[i + 1]], fill=path_color, width=path_width)

# Add final circle
draw.ellipse([points[-1][0] - path_width//2, points[-1][1] - path_width//2,
              points[-1][0] + path_width//2, points[-1][1] + path_width//2],
             fill=path_color)

# Add a red ball at start
start_x, start_y = points[0]
ball_size = 40
draw.ellipse([start_x - ball_size//2, start_y - ball_size//2,
              start_x + ball_size//2, start_y + ball_size//2],
             fill=(255, 100, 100), outline=(200, 50, 50), width=3)

# Add a flag at end
end_x, end_y = points[-1]
# Flag pole
draw.rectangle([end_x - 3, end_y - 60, end_x + 3, end_y], fill=(139, 69, 19))
# Flag
flag_points = [(end_x + 3, end_y - 60), (end_x + 40, end_y - 45), (end_x + 3, end_y - 30)]
draw.polygon(flag_points, fill=(255, 0, 0))

# Add some clouds for more natural look
for i in range(3):
    cloud_x = 200 + i * 350
    cloud_y = 50 + i * 30
    for j in range(3):
        draw.ellipse([cloud_x + j * 20 - 30, cloud_y - 20,
                      cloud_x + j * 20 + 30, cloud_y + 20],
                     fill=(255, 255, 255, 200))

# Save the image
image.save('test_photorealistic.png')
print("Created test_photorealistic.png")
