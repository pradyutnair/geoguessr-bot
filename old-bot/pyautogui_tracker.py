"""
PyAutoGUI-based Results Tracker for GeoGuessr Bot.

Uses PyAutoGUI to click the correct location marker and extract coordinates.
NO Selenium, NO Playwright, NO remote debugging - just screen automation!
"""

import csv
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import pyautogui
from PIL import Image


@dataclass
class RoundResult:
    """Results for a single round."""
    round_num: int
    predicted_lat: float
    predicted_lng: float
    true_lat: Optional[float] = None
    true_lng: Optional[float] = None
    distance_km: Optional[float] = None
    score: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_baseline: bool = False  # True if this is a baseline model prediction
    human_distance_km: Optional[float] = None  # Human player's distance error (km)
    human_score: Optional[int] = None  # Human player's score


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in kilometers."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def calculate_geoguessr_score(distance_km: float, max_distance_km: float = 14916.862) -> int:
    """
    Calculate GeoGuessr-style score using the non-linear decay formula.
    
    Score = 5000 * exp(-10 * (d / D))
    
    Where:
        d = distance between guess and true location (km)
        D = max distance (map diagonal, default is world map ~14917 km)
    
    Args:
        distance_km: Distance between guess and true location in km
        max_distance_km: Maximum possible distance (map diagonal). 
                         Default 14916.862 km is Earth's half circumference (world map).
    
    Returns:
        Score between 0 and 5000
    """
    if distance_km <= 0:
        return 5000
    
    score = 5000 * math.exp(-10 * (distance_km / max_distance_km))
    return int(round(score))


class PyAutoGUIResultsTracker:
    """
    Track GeoGuessr results using PyAutoGUI.
    
    This tracker:
    1. Waits for the results screen after a guess
    2. Finds the correct location marker using image recognition  
    3. Clicks on it to open Google Maps in a new tab
    4. Gets the URL from the address bar using xdotool
    5. Extracts coordinates from the URL
    6. Closes the tab and returns to the game
    """
    
    def __init__(self, output_dir: str = "results", stage1_checkpoint: str = None, stage2_checkpoint: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[RoundResult] = []
        self.connected = True  # Always "connected" since we use PyAutoGUI
        # Track true locations per round (shared across models)
        self.true_locations: dict = {}  # round_num -> (lat, lng)

        # Checkpoint information
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint

        # Session timestamp for CSV filename
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"game_results_{self.session_timestamp}.csv"

        # Path to marker image for recognition
        self.marker_image_path = Path(__file__).parent / "marker_icon.png"

        # Initialize CSV with headers
        self._init_csv()

        print(f"📊 PyAutoGUI Results Tracker initialized")
        print(f"   Output: {self.csv_path}")
        if self.stage1_checkpoint:
            print(f"   Stage1 checkpoint: {self.stage1_checkpoint}")
        if self.stage2_checkpoint:
            print(f"   Stage2 checkpoint: {self.stage2_checkpoint}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp',
                'predicted_lat', 'predicted_lng',
                'true_lat', 'true_lng',
                'distance_km', 'score',
                'stage1_checkpoint', 'stage2_checkpoint', 'model',
                'human_distance_km', 'human_score'
            ])
    
    def connect(self) -> bool:
        """No connection needed for PyAutoGUI - always returns True."""
        print("✅ PyAutoGUI tracker ready (no browser connection needed)")
        return True
    
    def record_prediction(self, round_num: int, pred_lat: float, pred_lng: float, is_baseline: bool = False):
        """Record a prediction (called when bot makes a guess)."""
        result = RoundResult(
            round_num=round_num,
            predicted_lat=pred_lat,
            predicted_lng=pred_lng,
            is_baseline=is_baseline
        )
        self.results.append(result)
        model_name = "baseline" if is_baseline else "main"
        print(f"📝 Recorded {model_name} prediction for round {round_num}: ({pred_lat:.4f}, {pred_lng:.4f})")
    
    def _find_marker_by_image(self, search_region: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int]]:
        """
        Find the correct location marker using image template matching.
        """
        marker_path = Path(__file__).parent / "marker_template.png"
        
        if not marker_path.exists():
            return None
        
        # Try different confidence levels
        for conf in [0.7, 0.6, 0.5]:
            try:
                location = pyautogui.locateOnScreen(
                    str(marker_path), 
                    confidence=conf,
                    region=search_region  # Only search in this region!
                )
                if location:
                    center = pyautogui.center(location)
                    print(f"      Image match found at confidence {conf}")
                    return (int(center.x), int(center.y))
            except pyautogui.ImageNotFoundException:
                continue
            except Exception as e:
                print(f"      Image search error: {e}")
                continue
        
        return None
    
    def _find_marker_by_color(self, search_region: Tuple[int, int, int, int] = None) -> Optional[Tuple[int, int]]:
        """
        Find the correct location marker - dark circle with yellow flag icon.
        """
        screenshot = pyautogui.screenshot()
        pixels = screenshot.load()
        width, height = screenshot.size
        
        if search_region:
            search_left, search_top, search_right, search_bottom = search_region
        else:
            search_left = int(width * 0.2)
            search_top = int(height * 0.2)
            search_right = int(width * 0.8)
            search_bottom = int(height * 0.8)
        
        candidates = []
        
        # Look for yellow flag color (high red, high green, low blue)
        for y in range(search_top, search_bottom, 2):
            for x in range(search_left, search_right, 2):
                r, g, b = pixels[x, y]
                
                # Yellow/gold flag: ~RGB(230, 195, 75)
                if r > 200 and g > 170 and b < 120 and r > b + 80:
                    # Check surrounding area for dark circle background
                    dark_count = 0
                    yellow_count = 0
                    
                    for dy in range(-20, 21, 2):
                        for dx in range(-20, 21, 2):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                pr, pg, pb = pixels[nx, ny]
                                # Dark navy/black background (~RGB(30, 35, 50))
                                if pr < 60 and pg < 60 and pb < 80:
                                    dark_count += 1
                                # Yellow flag
                                if pr > 200 and pg > 170 and pb < 120:
                                    yellow_count += 1
                    
                    # Must have dark background AND yellow pixels
                    if dark_count >= 10 and yellow_count >= 3:
                        score = dark_count + yellow_count * 3
                        candidates.append((x, y, score))
        
        if candidates:
            candidates.sort(key=lambda c: c[2], reverse=True)
            best = candidates[0]
            print(f"      Found {len(candidates)} candidates, best at ({best[0]}, {best[1]}) score={best[2]}")
            return (best[0], best[1])
        
        return None
    
    def _get_url_from_browser(self) -> Optional[str]:
        """
        Get the current URL from the browser address bar.
        Uses Ctrl+L to focus address bar, Ctrl+C to copy, then reads clipboard.
        """
        # Focus address bar
        pyautogui.hotkey('ctrl', 'l')
        time.sleep(0.15)
        
        # Copy URL
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(0.15)
        
        # Read clipboard using xclip
        result = subprocess.run(
            ['xclip', '-selection', 'clipboard', '-o'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        url = result.stdout.strip()
        
        # Escape to deselect
        pyautogui.press('escape')
        
        return url if url.startswith('http') else None
    
    def _parse_coords_from_url(self, url: str) -> Optional[Tuple[float, float]]:
        """Parse coordinates from Google Maps URL."""
        import urllib.parse
        
        # URL-decode first (handles %2C -> comma, etc.)
        url = urllib.parse.unquote(url)
        
        # Method 1: viewpoint parameter (GeoGuessr uses this)
        if 'viewpoint=' in url:
            match = re.search(r'viewpoint=(-?\d+\.?\d*),(-?\d+\.?\d*)', url)
            if match:
                lat = float(match.group(1))
                lng = float(match.group(2))
                return (lat, lng)
        
        # Method 2: @lat,lng in URL path
        match = re.search(r'@(-?\d+\.?\d*),(-?\d+\.?\d*)', url)
        if match:
            lat = float(match.group(1))
            lng = float(match.group(2))
            return (lat, lng)
        
        # Method 3: q= parameter
        match = re.search(r'[?&]q=(-?\d+\.?\d*),(-?\d+\.?\d*)', url)
        if match:
            lat = float(match.group(1))
            lng = float(match.group(2))
            return (lat, lng)
        
        return None
    
    def _click_marker_with_pyautogui(self, screen_regions: dict) -> Optional[Tuple[float, float]]:
        """
        Click the correct location marker and extract coordinates from Google Maps URL.
        """
        print(f"   ⏳ Looking for correct location marker...")
        
        # Wait for results screen to fully render
        time.sleep(1.5)
        
        # Get game region - YOUR GAME IS ON RIGHT MONITOR
        game_left = screen_regions.get("screen_top_left", [1935, 0])[0]
        game_top = screen_regions.get("screen_top_left", [1935, 0])[1]
        game_right = screen_regions.get("screen_bot_right", [3840, 1080])[0]
        game_bottom = screen_regions.get("screen_bot_right", [3840, 1080])[1]
        game_width = game_right - game_left
        game_height = game_bottom - game_top
        
        print(f"      Game region: ({game_left}, {game_top}) to ({game_right}, {game_bottom})")
        
        # Take screenshot of ONLY the game region
        game_screenshot = pyautogui.screenshot(region=(game_left, game_top, game_width, game_height))
        pixels = game_screenshot.load()
        
        # Find marker by color in this cropped screenshot
        # Marker is yellow flag (~RGB 230,195,75) on dark background (~RGB 30,35,50)
        marker_local_pos = None
        candidates = []
        
        for y in range(0, game_height, 3):
            for x in range(0, game_width, 3):
                r, g, b = pixels[x, y]
                
                # Yellow flag color
                if r > 200 and g > 170 and b < 100 and r > b + 100:
                    # Check for dark background nearby
                    dark_count = 0
                    yellow_count = 0
                    for dy in range(-15, 16, 3):
                        for dx in range(-15, 16, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < game_width and 0 <= ny < game_height:
                                pr, pg, pb = pixels[nx, ny]
                                if pr < 60 and pg < 60 and pb < 80:
                                    dark_count += 1
                                if pr > 200 and pg > 170 and pb < 100:
                                    yellow_count += 1
                    
                    if dark_count >= 8 and yellow_count >= 3:
                        candidates.append((x, y, dark_count + yellow_count))
        
        if candidates:
            candidates.sort(key=lambda c: c[2], reverse=True)
            marker_local_pos = (candidates[0][0], candidates[0][1])
            print(f"      Found {len(candidates)} candidates in game region")
        
        if not marker_local_pos:
            print("   ❌ Could not find marker in game region")
            return None
        
        # Convert local coordinates to absolute screen coordinates
        marker_x = game_left + marker_local_pos[0]
        marker_y = game_top + marker_local_pos[1]
        
        print(f"   ✅ Found marker at ({marker_x}, {marker_y}) [local: {marker_local_pos}]")
        
        # Save debug image
        from PIL import ImageDraw
        debug_img = game_screenshot.copy()
        draw = ImageDraw.Draw(debug_img)
        lx, ly = marker_local_pos
        draw.ellipse([lx-15, ly-15, lx+15, ly+15], outline="red", width=3)
        debug_img.save(Path(__file__).parent / "debug_marker_click.png")
        print(f"      Debug saved to debug_marker_click.png")
        
        # Click at absolute position
        print(f"   🖱️  Clicking at ({marker_x}, {marker_y})...")
        pyautogui.click(marker_x, marker_y)
        time.sleep(2.5)
        
        # Check URL
        url = self._get_url_from_browser()
        if not url:
            print("   ❌ Could not get URL")
            return None
        
        print(f"   🔗 URL: {url[:80]}...")
        
        # If Google Maps opened, extract coords
        if 'google.com/maps' in url or 'viewpoint=' in url:
            coords = self._parse_coords_from_url(url)
            pyautogui.hotkey('ctrl', 'w')  # Close tab
            time.sleep(0.3)
            if coords:
                print(f"   ✅ Coordinates: ({coords[0]:.6f}, {coords[1]:.6f})")
            return coords
        
        # Still on GeoGuessr - try clicking around the marker
        print("   ⚠️ Click didn't open Google Maps, trying nearby positions...")
        for dx, dy in [(0, 0), (0, -20), (0, 20), (-20, 0), (20, 0)]:
            x, y = marker_x + dx, marker_y + dy
            pyautogui.click(x, y)
            time.sleep(1.5)
            
            url = self._get_url_from_browser()
            if url and ('google.com/maps' in url or 'viewpoint=' in url):
                print(f"   ✅ Worked with offset ({dx}, {dy})")
                coords = self._parse_coords_from_url(url)
                pyautogui.hotkey('ctrl', 'w')
                time.sleep(0.3)
                return coords
        
        print("   ❌ Could not open Google Maps link")
        return None
    
    def capture_round_result(self, round_num: int, screen_regions: dict = None, human_mode: bool = False) -> Optional[RoundResult]:
        """
        Capture the true location by clicking the correct location marker.
        Calculates distance/score for all predictions for this round (main and baseline).
        If human_mode is True, prompts user for their distance and score.
        """
        # Find all predictions for this round
        round_predictions = [r for r in self.results if r.round_num == round_num]
        
        if not round_predictions:
            print(f"⚠️ No predictions recorded for round {round_num}")
            return None
        
        # Default screen regions if not provided
        if screen_regions is None:
            screen_regions = {
                "map_top_left_1": [1500, 400],
                "map_bot_right_1": [1900, 800]
            }
        
        # Extract coordinates from marker (only once per round)
        true_coords = None
        if round_num not in self.true_locations:
            coords = self._click_marker_with_pyautogui(screen_regions)
            if coords:
                self.true_locations[round_num] = coords
                true_coords = coords
            else:
                print(f"⚠️ Could not extract true location for round {round_num}")
        else:
            true_coords = self.true_locations[round_num]
        
        # Calculate distance and score for each prediction
        if true_coords:
            true_lat, true_lng = true_coords
            
            print(f"\n📊 Round {round_num} Results:")
            print(f"   True location: ({true_lat:.4f}, {true_lng:.4f})")
            
            # Process each prediction (main and baseline)
            for result in round_predictions:
                result.true_lat = true_lat
                result.true_lng = true_lng
                
                # Calculate distance
                result.distance_km = haversine_distance(
                    result.predicted_lat, result.predicted_lng,
                    result.true_lat, result.true_lng
                )
                
                # Calculate GeoGuessr-style score
                result.score = calculate_geoguessr_score(result.distance_km)
                
                model_name = "baseline" if result.is_baseline else "main"
                print(f"   {model_name.capitalize()} model:")
                print(f"      Distance: {result.distance_km:.1f} km")
                print(f"      Score: {result.score}")
                
                # Write to CSV
                self._append_to_csv(result)
            
            # If human mode, prompt for human input
            if human_mode:
                self._prompt_human_input(round_predictions[0], round_num)
            
            # Return the first result (main model) for backward compatibility
            main_result = next((r for r in round_predictions if not r.is_baseline), round_predictions[0])
            return main_result
        else:
            # Still write predictions to CSV even if true location is missing
            for result in round_predictions:
                self._append_to_csv(result)
            
            # If human mode, still prompt for human input
            if human_mode:
                self._prompt_human_input(round_predictions[0], round_num)
            
            return round_predictions[0]
    
    def _append_to_csv(self, result: RoundResult):
        """Append a result to the CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                result.round_num,
                result.timestamp,
                f"{result.predicted_lat:.6f}",
                f"{result.predicted_lng:.6f}",
                f"{result.true_lat:.6f}" if result.true_lat is not None else "",
                f"{result.true_lng:.6f}" if result.true_lng is not None else "",
                f"{result.distance_km:.2f}" if result.distance_km is not None else "",
                result.score if result.score else "",
                self.stage1_checkpoint or "",
                self.stage2_checkpoint or "",
                "baseline" if result.is_baseline else "main",  # Add model identifier
                f"{result.human_distance_km:.2f}" if result.human_distance_km is not None else "",
                result.human_score if result.human_score else ""
            ])
    
    def _prompt_human_input(self, result: RoundResult, round_num: int):
        """Prompt user for their distance error and score, then update and write to CSV."""
        print(f"\n👤 HUMAN INPUT for Round {round_num}:")
        
        # Get distance input
        while True:
            try:
                distance_input = input("   Enter your distance error (km): ").strip()
                if distance_input:
                    human_distance = float(distance_input)
                    if human_distance < 0:
                        print("   ⚠️ Distance must be non-negative. Please try again.")
                        continue
                    result.human_distance_km = human_distance
                    break
                else:
                    print("   ⚠️ Please enter a distance value.")
                    continue
            except ValueError:
                print("   ⚠️ Invalid input. Please enter a number.")
                continue
        
        # Get score input
        while True:
            try:
                score_input = input("   Enter your score: ").strip()
                if score_input:
                    human_score = int(score_input)
                    if human_score < 0 or human_score > 5000:
                        print("   ⚠️ Score should be between 0 and 5000. Please try again.")
                        continue
                    result.human_score = human_score
                    break
                else:
                    print("   ⚠️ Please enter a score value.")
                    continue
            except ValueError:
                print("   ⚠️ Invalid input. Please enter an integer.")
                continue
        
        print(f"   ✅ Recorded: Distance = {result.human_distance_km:.1f} km, Score = {result.human_score}")
        
        # Update CSV with human data - need to update all rows for this round
        self._update_csv_with_human_data(round_num, result.human_distance_km, result.human_score)
    
    def _update_csv_with_human_data(self, round_num: int, human_distance_km: float, human_score: int):
        """Update CSV rows for a specific round with human data."""
        # Read all rows
        rows = []
        with open(self.csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            rows.append(header)
            for row in reader:
                if len(row) > 0 and row[0] == str(round_num):
                    # Update human columns (indices 11 and 12)
                    if len(row) >= 13:
                        row[11] = f"{human_distance_km:.2f}"
                        row[12] = str(human_score)
                    elif len(row) == 12:
                        # Old format without human columns, add them
                        row.append(f"{human_distance_km:.2f}")
                        row.append(str(human_score))
                rows.append(row)
        
        # Write back
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def get_summary(self) -> dict:
        """Get summary statistics for all recorded rounds."""
        if not self.results:
            return {}
        
        # Count unique rounds (not total results, since comparison mode has 2 per round)
        unique_rounds = set(r.round_num for r in self.results)
        
        distances = [r.distance_km for r in self.results if r.distance_km is not None]
        scores = [r.score for r in self.results if r.score is not None]
        
        # Separate main and baseline stats if in comparison mode
        main_distances = [r.distance_km for r in self.results if r.distance_km is not None and not r.is_baseline]
        baseline_distances = [r.distance_km for r in self.results if r.distance_km is not None and r.is_baseline]
        main_scores = [r.score for r in self.results if r.score is not None and not r.is_baseline]
        baseline_scores = [r.score for r in self.results if r.score is not None and r.is_baseline]
        
        summary = {
            "total_rounds": len(unique_rounds),
            "rounds_with_results": len(set(r.round_num for r in self.results if r.distance_km is not None)),
        }
        
        if distances:
            summary["avg_distance_km"] = sum(distances) / len(distances)
            summary["min_distance_km"] = min(distances)
            summary["max_distance_km"] = max(distances)
        
        if scores:
            summary["total_score"] = sum(scores)
            summary["avg_score"] = sum(scores) / len(scores)
        
        # Add comparison stats if baseline exists
        if baseline_distances:
            summary["main_avg_distance_km"] = sum(main_distances) / len(main_distances) if main_distances else None
            summary["baseline_avg_distance_km"] = sum(baseline_distances) / len(baseline_distances)
            summary["main_avg_score"] = sum(main_scores) / len(main_scores) if main_scores else None
            summary["baseline_avg_score"] = sum(baseline_scores) / len(baseline_scores)
        
        return summary
    
    def print_summary(self):
        """Print game summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("📊 GAME SUMMARY")
        print("="*60)
        
        if not summary:
            print("No results recorded.")
            return
        
        print(f"Rounds played: {summary['total_rounds']}")
        print(f"Rounds with results: {summary['rounds_with_results']}")
        
        # Check if comparison mode was used
        if 'baseline_avg_distance_km' in summary:
            print(f"\n📊 Comparison Mode Results:")
            print(f"\nMain Model:")
            if 'main_avg_distance_km' in summary and summary['main_avg_distance_km'] is not None:
                print(f"   Average distance: {summary['main_avg_distance_km']:.1f} km")
            if 'main_avg_score' in summary and summary['main_avg_score'] is not None:
                print(f"   Average score: {summary['main_avg_score']:.0f}")
            
            print(f"\nBaseline Model:")
            print(f"   Average distance: {summary['baseline_avg_distance_km']:.1f} km")
            print(f"   Average score: {summary['baseline_avg_score']:.0f}")
        else:
            # Single model mode
            if 'avg_distance_km' in summary:
                print(f"\nDistance Statistics:")
                print(f"   Average: {summary['avg_distance_km']:.1f} km")
                print(f"   Best: {summary['min_distance_km']:.1f} km")
                print(f"   Worst: {summary['max_distance_km']:.1f} km")
            
            if 'total_score' in summary:
                print(f"\nScore Statistics:")
                print(f"   Total: {summary['total_score']}")
                print(f"   Average: {summary['avg_score']:.0f}")
        
        print(f"\nResults saved to: {self.csv_path}")
        print("="*60)
    
    def close(self):
        """Clean up - nothing to do for PyAutoGUI tracker."""
        pass


# Alias for compatibility
ResultsTracker = PyAutoGUIResultsTracker
