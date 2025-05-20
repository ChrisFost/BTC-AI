#!/usr/bin/env python
"""
Create a dashboard preview image for the Progressive Training UI.
This script generates a visual preview of what the dashboard will look like.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import importlib
from PIL import Image, ImageDraw

# Set up styling
sns.set_style("darkgrid")
sns.set_context("talk")

def create_dashboard_preview():
    """Create a preview of the dashboard to show in the UI"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dashboard_preview.png")
    
    # Create figure with complex grid
    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[2, 1, 1])
    
    # Create sample data
    episodes = np.arange(100)
    scalping_rewards = 0.2 + 0.4 * np.cumsum(0.01 * np.random.randn(100))
    short_rewards = 0.1 + 0.3 * np.cumsum(0.01 * np.random.randn(100))
    medium_rewards = 0.05 + 0.2 * np.cumsum(0.01 * np.random.randn(100))
    
    # Training progress subplot (top left)
    ax_progress = fig.add_subplot(gs[0, 0])
    ax_progress.plot(episodes, scalping_rewards, label="Scalping", color="purple")
    ax_progress.plot(episodes, short_rewards, label="Short", color="blue")
    ax_progress.plot(episodes, medium_rewards, label="Medium", color="green")
    ax_progress.set_title("Training Progress")
    ax_progress.set_ylabel("Reward")
    ax_progress.legend()
    ax_progress.grid(True, alpha=0.3)
    
    # Knowledge transfer subplot (top middle)
    ax_transfer = fig.add_subplot(gs[0, 1])
    buckets = ["Scalping", "Short", "Medium", "Long"]
    bucket_positions = {bucket: i for i, bucket in enumerate(buckets)}
    
    # Create transfer markers
    transfers = [
        {"source": "Scalping", "target": "Short", "episode": 25, "type": "weights"},
        {"source": "Scalping", "target": "Medium", "episode": 30, "type": "features"},
        {"source": "Short", "target": "Medium", "episode": 55, "type": "horizons"},
        {"source": "Medium", "target": "Long", "episode": 85, "type": "weights"}
    ]
    
    # Plot transfer events
    for transfer in transfers:
        source_pos = bucket_positions[transfer["source"]]
        target_pos = bucket_positions[transfer["target"]]
        episode = transfer["episode"]
        
        # Set marker based on transfer type
        if transfer["type"] == "weights":
            marker = "o"
        elif transfer["type"] == "features":
            marker = "s"
        else:
            marker = "^"
        
        # Plot the transfer
        ax_transfer.plot([episode, episode], [source_pos, target_pos], 'k-', alpha=0.5)
        ax_transfer.plot(episode, source_pos, marker, color='blue', markersize=8)
        ax_transfer.plot(episode, target_pos, marker, color='green', markersize=8)
    
    ax_transfer.set_yticks(range(len(buckets)))
    ax_transfer.set_yticklabels(buckets)
    ax_transfer.set_title("Knowledge Transfer")
    ax_transfer.set_xlabel("Episode")
    ax_transfer.grid(True, alpha=0.3)
    
    # Feature importance subplot (top right)
    ax_features = fig.add_subplot(gs[0, 2])
    features = ["price", "volume", "macd", "rsi", "ema", "bollinger"]
    importance = [0.8, 0.6, 0.5, 0.7, 0.4, 0.3]
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    
    bars = ax_features.barh(features, importance, color=colors)
    ax_features.set_title("Feature Importance")
    ax_features.set_xlim(0, 1)
    ax_features.grid(True, alpha=0.3)
    
    # Memory usage subplot (middle row, spans all columns)
    ax_memory = fig.add_subplot(gs[1, :])
    time_steps = np.arange(100)
    memory_usage = 20 + 5 * np.sin(time_steps/10) + np.cumsum(0.1 * np.random.randn(100))
    memory_usage = np.clip(memory_usage, 0, 100)
    
    ax_memory.plot(time_steps, memory_usage, 'r-')
    ax_memory.axhline(y=80, color='r', linestyle='--', alpha=0.5)
    ax_memory.fill_between(time_steps, memory_usage, color='r', alpha=0.2)
    ax_memory.set_title("Memory Usage")
    ax_memory.set_ylabel("Usage (%)")
    ax_memory.set_ylim(0, 100)
    ax_memory.grid(True, alpha=0.3)
    
    # Add transfer event markers to memory plot
    for transfer in transfers:
        ax_memory.axvline(x=transfer["episode"], color='b', linestyle=':', alpha=0.7)
    
    # Log content (bottom left, spans 2 columns)
    ax_log = fig.add_subplot(gs[2, :2])
    ax_log.text(0.5, 0.5, "Training Log\n\n[INFO] Starting training for Scalping bucket\n[INFO] Episode 10/100 completed\n[INFO] Transferring knowledge to Short bucket", 
               ha='center', va='center', fontfamily='monospace')
    ax_log.axis('off')
    
    # System metrics (bottom right)
    ax_metrics = fig.add_subplot(gs[2, 2])
    metrics = {
        "CPU Usage": "45%",
        "GPU Usage": "78%",
        "Memory": "4.2 GB",
        "Training Time": "01:23:45",
        "Win Rate": "62%"
    }
    
    y_pos = np.arange(len(metrics))
    ax_metrics.axis('off')
    
    # Add metric text
    for i, (metric, value) in enumerate(metrics.items()):
        ax_metrics.text(0.1, 0.9 - (i * 0.15), f"{metric}:", ha='left', va='center', fontweight='bold')
        ax_metrics.text(0.6, 0.9 - (i * 0.15), f"{value}", ha='left', va='center')
    
    # Final formatting
    plt.tight_layout()
    plt.suptitle("Progressive Training Dashboard", fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)
    
    # Save the preview
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Preview image saved to: {output_path}")
    plt.close(fig)
    
    # Also create an icon for the application
    create_icon(output_dir)
    
    return output_path

def create_icon(output_dir):
    """Create an icon for the application"""
    # Create a simple icon
    icon_size = 256
    icon_path = os.path.join(output_dir, "icon.ico")
    
    # Skip if icon already exists
    if os.path.exists(icon_path):
        return
    
    try:
        # Try to create the icon
        # First create a PNG file
        img = Image.new("RGBA", (icon_size, icon_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw a purple circle for the background
        circle_color = (120, 81, 169, 255)  # Purple
        draw.ellipse((0, 0, icon_size, icon_size), fill=circle_color)
        
        # Draw a smaller white circle in the middle
        inner_margin = icon_size // 4
        draw.ellipse((inner_margin, inner_margin, 
                     icon_size - inner_margin, icon_size - inner_margin), 
                   fill=(255, 255, 255, 255))
        
        # Draw a stylized "M" in the middle
        draw.polygon([
            (icon_size // 3, icon_size // 3),
            (icon_size // 2, icon_size * 2 // 3),
            (icon_size * 2 // 3, icon_size // 3),
            (icon_size * 3 // 4, icon_size // 2),
            (icon_size * 2 // 3, icon_size * 2 // 3),
            (icon_size // 2, icon_size // 2),
            (icon_size // 3, icon_size * 2 // 3),
            (icon_size // 4, icon_size // 2),
        ], fill=circle_color)
        
        # Save as PNG
        png_path = os.path.join(output_dir, "icon.png")
        img.save(png_path)
        
        # Try to convert to ico if PIL supports it
        try:
            img.save(icon_path, format="ICO")
        except:
            # If PIL doesn't support ICO, try using other methods
            try:
                import win32api
                import win32con
                import win32ui
                
                # Create a Windows icon using win32 APIs
                ico_sizes = [(icon_size, icon_size)]
                icon_images = [(png_path, win32con.LR_LOADFROMFILE)]
                
                win32ui.CreateWindowFromHandle(win32api.GetDesktopWindow())
                hicon = win32ui.CreateIcon(icon_images)
                
                with open(icon_path, "wb") as f:
                    hicon.SaveICO(f)
            except:
                # If all conversion attempts fail, just use the PNG
                print(f"Icon creation failed, but PNG icon saved to: {png_path}")
    
    except Exception as e:
        print(f"Error creating icon: {e}")
        # Icon is not critical, so just continue

if __name__ == "__main__":
    create_dashboard_preview() 