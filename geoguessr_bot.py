"""
GeoGuessr Bot using PyAutoGUI and Stage 2 ML model API.

Takes screenshots, sends to API server, clicks on predicted location on minimap.
Concept logging happens server-side automatically.
Results tracking captures true locations and scores.
"""

import base64
import math
import os
import requests
from io import BytesIO
from time import sleep
from typing import Tuple, Optional

import pyautogui
from PIL import Image

from pyautogui_tracker import PyAutoGUIResultsTracker as ResultsTracker


class GeoBot:
    """Bot that plays GeoGuessr using ML model predictions."""
    
    def __init__(
        self, 
        screen_regions: dict, 
        player: int = 1,
        api_url: str = "http://127.0.0.1:5000/api/v1/predict",
    ):
        self.player = player
        self.screen_regions = screen_regions
        self.api_url = api_url
        
        # Screen region for capturing the panorama view
        self.screen_x, self.screen_y = screen_regions["screen_top_left"]
        self.screen_w = screen_regions["screen_bot_right"][0] - self.screen_x
        self.screen_h = screen_regions["screen_bot_right"][1] - self.screen_y
        self.screen_xywh = (self.screen_x, self.screen_y, self.screen_w, self.screen_h)

        # Minimap region
        self.map_x, self.map_y = screen_regions[f"map_top_left_{player}"]
        self.map_w = screen_regions[f"map_bot_right_{player}"][0] - self.map_x
        self.map_h = screen_regions[f"map_bot_right_{player}"][1] - self.map_y
        self.minimap_xywh = (self.map_x, self.map_y, self.map_w, self.map_h)

        # Button locations
        self.next_round_button = screen_regions.get("next_round_button")
        self.confirm_button = screen_regions[f"confirm_button_{player}"]
        
        # Calculate calibration from reference points
        self.MAP_CALIBRATION = self._calculate_calibration()
        
        print(f"🤖 GeoBot initialized")
        print(f"   📍 API: {api_url}")
        print(f"   📺 Screen region: {self.screen_w}x{self.screen_h}")
        print(f"   🗺️  Minimap region: {self.map_w}x{self.map_h}")
        print(f"   📐 Calibration: {self.MAP_CALIBRATION}")
    
    def _calculate_calibration(self) -> dict:
        """
        Calculate calibration parameters from reference points (Kodiak, Hobart).
        
        Returns dict with x_offset, y_offset, x_scale, y_scale
        """
        # Default calibration (fallback if no reference points)
        # These values work well for GeoGuessr's default world minimap
        # The minimap has some padding/margins that reduce the effective area
        default_cal = {
            "x_offset": 0.02,  # Small left margin
            "y_offset": 0.05,  # Top margin (for UI elements)
            "x_scale": 0.96,   # Effective width is ~96% of minimap
            "y_scale": 0.90,   # Effective height is ~90% (top/bottom UI)
        }
        
        # Check if calibration points exist
        kodiak = self.screen_regions.get("calibration_kodiak")
        hobart = self.screen_regions.get("calibration_hobart")
        
        if not kodiak or not hobart:
            print("   ⚠️  No calibration points found, using default calibration")
            return default_cal
        
        # Get theoretical Mercator ratios for reference points
        kodiak_lat, kodiak_lng = kodiak["lat"], kodiak["lng"]
        hobart_lat, hobart_lng = hobart["lat"], hobart["lng"]
        
        kodiak_x_ratio, kodiak_y_ratio = self.lat_lng_to_mercator(kodiak_lat, kodiak_lng)
        hobart_x_ratio, hobart_y_ratio = self.lat_lng_to_mercator(hobart_lat, hobart_lng)
        
        # Get actual pixel positions (relative to minimap)
        kodiak_pixel = kodiak["pixel"]
        hobart_pixel = hobart["pixel"]
        
        kodiak_pixel_x = kodiak_pixel[0] - self.map_x
        kodiak_pixel_y = kodiak_pixel[1] - self.map_y
        hobart_pixel_x = hobart_pixel[0] - self.map_x
        hobart_pixel_y = hobart_pixel[1] - self.map_y
        
        # Convert to pixel ratios [0, 1]
        kodiak_pixel_x_ratio = kodiak_pixel_x / self.map_w
        kodiak_pixel_y_ratio = kodiak_pixel_y / self.map_h
        hobart_pixel_x_ratio = hobart_pixel_x / self.map_w
        hobart_pixel_y_ratio = hobart_pixel_y / self.map_h
        
        # Calculate calibration: pixel_ratio = offset + mercator_ratio * scale
        # Using two points to solve for offset and scale:
        # For X: pixel_x = offset_x + mercator_x * scale_x
        # For Y: pixel_y = offset_y + mercator_y * scale_y
        
        # X calibration: solve system of equations
        # kodiak_px = offset_x + kodiak_mx * scale_x
        # hobart_px = offset_x + hobart_mx * scale_x
        # => scale_x = (kodiak_px - hobart_px) / (kodiak_mx - hobart_mx)
        # => offset_x = kodiak_px - kodiak_mx * scale_x
        
        mercator_x_diff = kodiak_x_ratio - hobart_x_ratio
        pixel_x_diff = kodiak_pixel_x_ratio - hobart_pixel_x_ratio
        
        if abs(mercator_x_diff) > 0.001:  # Avoid division by zero
            x_scale = pixel_x_diff / mercator_x_diff
            x_offset = kodiak_pixel_x_ratio - kodiak_x_ratio * x_scale
        else:
            x_scale = 1.0
            x_offset = 0.0
        
        # Y calibration: same approach
        mercator_y_diff = kodiak_y_ratio - hobart_y_ratio
        pixel_y_diff = kodiak_pixel_y_ratio - hobart_pixel_y_ratio
        
        if abs(mercator_y_diff) > 0.001:  # Avoid division by zero
            y_scale = pixel_y_diff / mercator_y_diff
            y_offset = kodiak_pixel_y_ratio - kodiak_y_ratio * y_scale
        else:
            y_scale = 1.0
            y_offset = 0.0
        
        calibration = {
            "x_offset": x_offset,
            "y_offset": y_offset,
            "x_scale": x_scale,
            "y_scale": y_scale,
        }
        
        print(f"   🎯 Calibration calculated from reference points:")
        print(f"      Kodiak: ({kodiak_lat:.2f}, {kodiak_lng:.2f}) -> pixel ({kodiak_pixel_x}, {kodiak_pixel_y})")
        print(f"      Hobart: ({hobart_lat:.2f}, {hobart_lng:.2f}) -> pixel ({hobart_pixel_x}, {hobart_pixel_y})")
        
        return calibration

    @staticmethod
    def pil_to_base64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def predict_location_from_api(image: Image.Image, api_url: str) -> Optional[Tuple[float, float]]:
        """Send screenshot to ML API at specified URL and return predicted lat/lng."""
        image_b64 = GeoBot.pil_to_base64(image)
        payload = {"image": f"data:image/png;base64,{image_b64}"}
        
        response = requests.post(
            api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"❌ API Error ({api_url}): {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        lat = result["results"]["lat"]
        lng = result["results"]["lng"]
        
        return lat, lng

    def predict_location(self, image: Image.Image) -> Optional[Tuple[float, float]]:
        """Send screenshot to ML API and return predicted lat/lng."""
        return self.predict_location_from_api(image, self.api_url)

    def lat_lng_to_mercator(self, lat: float, lng: float) -> Tuple[float, float]:
        """
        Convert lat/lng to normalized Web Mercator coordinates [0,1].
        
        Returns (x_ratio, y_ratio) where:
        - x_ratio: 0 = -180° longitude, 1 = +180° longitude
        - y_ratio: 0 = north (85°), 1 = south (-85°)
        """
        MAX_LAT = 85.051129
        lat = max(-MAX_LAT, min(MAX_LAT, lat))
        lng = ((lng + 180) % 360) - 180
        
        # X: Linear mapping of longitude
        x_ratio = (lng + 180) / 360.0
        
        # Y: Web Mercator projection
        lat_rad = math.radians(lat)
        mercator_y = math.log(math.tan(math.pi / 4 + lat_rad / 2))
        max_mercator = math.log(math.tan(math.pi / 4 + math.radians(MAX_LAT) / 2))
        y_ratio = (1 - mercator_y / max_mercator) / 2
        
        return x_ratio, y_ratio
    
    def mercator_to_screen(self, x_ratio: float, y_ratio: float, 
                           map_bounds: Tuple[float, float, float, float] = None) -> Tuple[int, int]:
        """
        Convert normalized mercator ratios to screen pixel coordinates.
        
        Args:
            x_ratio, y_ratio: Normalized [0,1] mercator coordinates
            map_bounds: (min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio) of visible map
                       If None, assumes full world view (0,0,1,1)
        """
        if map_bounds is None:
            map_bounds = (0.0, 0.0, 1.0, 1.0)
        
        min_x, min_y, max_x, max_y = map_bounds
        
        # Calculate position within visible bounds
        if max_x > min_x:
            pixel_x_ratio = (x_ratio - min_x) / (max_x - min_x)
        else:
            pixel_x_ratio = 0.5
            
        if max_y > min_y:
            pixel_y_ratio = (y_ratio - min_y) / (max_y - min_y)
        else:
            pixel_y_ratio = 0.5
        
        # Map to screen coordinates
        x = self.map_x + int(pixel_x_ratio * self.map_w)
        y = self.map_y + int(pixel_y_ratio * self.map_h)
        
        return x, y

    def lat_lon_to_screen_coords(self, lat: float, lng: float) -> Tuple[int, int]:
        """
        Convert latitude and longitude to pixel coordinates on the minimap.
        Assumes full world view (zoomed out completely).
        Applies calibration from reference points.
        """
        x_ratio, y_ratio = self.lat_lng_to_mercator(lat, lng)
        
        # Apply calibration
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + y_ratio * cal["y_scale"]
        
        # Clamp to valid range
        adjusted_x_ratio = max(0.0, min(1.0, adjusted_x_ratio))
        adjusted_y_ratio = max(0.0, min(1.0, adjusted_y_ratio))
        
        # Convert to screen coordinates
        x = self.map_x + int(adjusted_x_ratio * self.map_w)
        y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   Conversion: ({lat:.4f}, {lng:.4f}) -> pixel ({x}, {y})")
        print(f"   Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
        
        return x, y

    def clamp_to_minimap(self, x: int, y: int) -> Tuple[int, int]:
        """Clamp coordinates to be within the minimap bounds."""
        margin = 10
        x_clamped = max(self.map_x + margin, min(self.map_x + self.map_w - margin, x))
        y_clamped = max(self.map_y + margin, min(self.map_y + self.map_h - margin, y))
        
        if x != x_clamped or y != y_clamped:
            print(f"   ⚠️ Clamped from ({x}, {y}) to ({x_clamped}, {y_clamped})")
        
        return x_clamped, y_clamped

    def expand_minimap(self):
        """Hover over minimap to expand it."""
        hover_x = self.map_x + self.map_w - 20
        hover_y = self.map_y + self.map_h - 20
        pyautogui.moveTo(hover_x, hover_y, duration=0.3)
        sleep(0.8)

    def reset_minimap_zoom(self):
        """
        Zoom out the minimap fully to reset to world view.
        This gives us a known state to work from.
        """
        center_x = self.map_x + self.map_w // 2
        center_y = self.map_y + self.map_h // 2
        
        pyautogui.moveTo(center_x, center_y, duration=0.1)
        sleep(0.1)
        
        # Zoom out a lot to ensure we're at world view
        for _ in range(12):
            pyautogui.scroll(-5)  # Negative = zoom out
            sleep(0.05)
        
        sleep(0.4)
        print("   🌍 Reset minimap to world view")
    
    def smart_place_guess(self, lat: float, lng: float, debug: bool = True):
        """
        Precisely place a guess on the GeoGuessr minimap.
        
        Strategy:
        1. Expand minimap
        2. Zoom out fully to get world view (known state)
        3. Calculate where target is on world view
        4. Single-click to place marker at correct position
        5. Scroll-zoom to increase precision (cursor stays on marker)
        
        Key insight: Every click on the map MOVES the marker. So we must
        only click once at the correct position, then just zoom without clicking.
        """
        print(f"   🎯 Smart placement for ({lat:.4f}, {lng:.4f})")
        
        # Step 1: Ensure minimap is expanded
        self.expand_minimap()
        
        # Step 2: Reset to world view (fully zoomed out)
        self.reset_minimap_zoom()
        sleep(0.3)
        
        # Step 3: Calculate target position on world view
        target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
        
        # Apply calibration from class constants
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
        
        # Clamp ratios to valid range
        adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
        adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
        
        # Convert to screen coordinates
        target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
        target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   📍 Target position: ({target_x}, {target_y})")
        print(f"   📐 Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
        
        if debug:
            # Save debug screenshot showing where we intend to click
            from PIL import ImageDraw
            debug_img = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
            draw = ImageDraw.Draw(debug_img)
            local_x = target_x - self.map_x
            local_y = target_y - self.map_y
            draw.ellipse([local_x-10, local_y-10, local_x+10, local_y+10], outline="red", width=3)
            draw.line([local_x-15, local_y, local_x+15, local_y], fill="red", width=2)
            draw.line([local_x, local_y-15, local_x, local_y+15], fill="red", width=2)
            debug_img.save("debug_target_position.png")
            print(f"   📸 Debug image saved to debug_target_position.png")
        
        # Step 4: Move to target and place marker with single click
        print("   🎯 Placing marker at target position")
        pyautogui.moveTo(target_x, target_y, duration=0.15)
        sleep(0.1)
        pyautogui.click(target_x, target_y)  # Single click places marker
        sleep(0.3)
        
        # Step 5: Scroll-zoom for precision (cursor stays on marker)
        print("   🔍 Scroll-zooming for precision...")
        for i in range(10):
            pyautogui.scroll(4)  # Zoom in
            sleep(0.08)
        sleep(0.3)
        
        # DO NOT click again - the marker is already correctly placed!
        print(f"   ✅ Marker placed - no additional clicks needed")

    def scroll_zoom_place_guess(self, lat: float, lng: float, debug: bool = True):
        """
        Improved method using iterative zoom-and-pan approach.
        
        Strategy:
        1. Reset to world view
        2. Calculate target position on world map
        3. Move to target, zoom in while keeping cursor on target
        4. The minimap will zoom centered on cursor position
        5. Click to place marker
        
        This approach is more robust because scroll-zoom typically
        keeps the point under cursor at the same location.
        """
        print(f"   🎯 Scroll-zoom placement for ({lat:.4f}, {lng:.4f})")
        
        # Step 1: Expand minimap
        self.expand_minimap()
        
        # Step 2: Reset to world view (fully zoom out)
        self.reset_minimap_zoom()
        sleep(0.3)  # Extra wait for zoom animation
        
        # Step 3: Calculate target position using improved method
        # Use simpler, more robust calculation
        target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
        
        # Apply calibration if available, otherwise use defaults
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
        
        # Clamp with slightly more margin (minimap edges are unreliable)
        adjusted_x_ratio = max(0.08, min(0.92, adjusted_x_ratio))
        adjusted_y_ratio = max(0.08, min(0.92, adjusted_y_ratio))
        
        target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
        target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   📍 Target position: ({target_x}, {target_y})")
        print(f"   📐 Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
        
        if debug:
            # Save debug screenshot showing calculated target
            from PIL import ImageDraw
            debug_img = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
            draw = ImageDraw.Draw(debug_img)
            local_x = target_x - self.map_x
            local_y = target_y - self.map_y
            # Draw crosshair at target
            draw.ellipse([local_x-10, local_y-10, local_x+10, local_y+10], outline="lime", width=3)
            draw.line([local_x-15, local_y, local_x+15, local_y], fill="lime", width=2)
            draw.line([local_x, local_y-15, local_x, local_y+15], fill="lime", width=2)
            debug_img.save("debug_scroll_target.png")
            print(f"   📸 Debug image saved to debug_scroll_target.png")
        
        # Step 4: Move to target position
        pyautogui.moveTo(target_x, target_y, duration=0.2)
        sleep(0.2)
        
        # Step 5: Scroll zoom in controlled increments
        # Use smaller zoom steps for more control
        print("   🔍 Scroll zooming on target (10 clicks in 2 phases)...")
        
        # Phase 1: Initial zoom (5 clicks)
        for i in range(5):
            pyautogui.scroll(4)  # Zoom in
            sleep(0.1)
        sleep(0.3)  # Wait for map to settle
        
        # Phase 2: Fine zoom (5 more clicks)
        for i in range(5):
            pyautogui.scroll(3)  # Smaller zoom increments
            sleep(0.1)
        sleep(0.3)
        
        # Step 6: Click at current cursor position to place marker
        current_x, current_y = pyautogui.position()
        print(f"   ✅ Clicking at cursor position ({current_x}, {current_y})")
        pyautogui.click(current_x, current_y)
        
        sleep(0.2)
    
    def iterative_pan_zoom_place(self, lat: float, lng: float, debug: bool = True):
        """
        Robust method: Click at target, then scroll-zoom for precision.
        
        This method works by:
        1. Calculating exact target position on world view
        2. Single-clicking to place marker at correct position
        3. Using scroll-zoom (cursor stays on marker) to zoom in
        4. NOT clicking again - marker is already in correct place
        
        Key insight: Every click on the map MOVES the marker. So we must
        only click once at the correct position, then just zoom without clicking.
        """
        print(f"   🎯 Pan-zoom placement for ({lat:.4f}, {lng:.4f})")
        
        # Step 1: Expand minimap
        self.expand_minimap()
        
        # Step 2: Reset to world view
        self.reset_minimap_zoom()
        sleep(0.4)
        
        # Step 3: Calculate target position on world view
        target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
        cal = self.MAP_CALIBRATION
        adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
        adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
        adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
        adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
        
        target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
        target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
        
        print(f"   📍 Target position: ({target_x}, {target_y})")
        
        if debug:
            from PIL import ImageDraw
            debug_img = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
            draw = ImageDraw.Draw(debug_img)
            local_x = target_x - self.map_x
            local_y = target_y - self.map_y
            draw.ellipse([local_x-10, local_y-10, local_x+10, local_y+10], outline="cyan", width=3)
            debug_img.save("debug_pan_zoom_target.png")
        
        # Step 4: Move to target and place marker with single click
        print("   🎯 Placing marker at target position")
        pyautogui.moveTo(target_x, target_y, duration=0.15)
        sleep(0.1)
        pyautogui.click(target_x, target_y)  # Single click places marker
        sleep(0.3)
        
        # Step 5: Zoom in with scroll (cursor stays on marker position)
        # Scroll-zoom keeps the point under cursor stationary
        print("   🔍 Scroll-zooming for precision (cursor on marker)")
        for i in range(8):
            pyautogui.scroll(4)  # Zoom in
            sleep(0.08)
        sleep(0.3)
        
        # DO NOT click again - the marker is already correctly placed!
        print(f"   ✅ Marker placed at ({target_x}, {target_y}) - no additional clicks needed")
    
    def zoom_minimap(self, x: int, y: int, zoom_clicks: int = 3):
        """
        Zoom into the minimap at the target location for more precise clicking.
        
        Args:
            x, y: Target screen coordinates to zoom towards
            zoom_clicks: Number of scroll wheel clicks to zoom in
        """
        # Move to target location on minimap
        pyautogui.moveTo(x, y, duration=0.2)
        sleep(0.2)
        
        # Scroll to zoom in (positive = zoom in on most maps)
        for _ in range(zoom_clicks):
            pyautogui.scroll(3)  # Scroll up to zoom in
            sleep(0.15)
        
        sleep(0.3)  # Wait for zoom animation
        
    def click_on_map_precise(self, lat: float, lng: float, use_zoom: bool = True, method: str = "scroll"):
        """
        Click on the minimap with zoom-based precision.
        
        Args:
            lat, lng: Target coordinates
            use_zoom: Whether to use zoom for better precision
            method: 
                - "scroll": Scroll zoom, keeps cursor on target (default)
                - "doubleclick": Double-click to zoom and center
                - "panzoom": Iterative pan-zoom (most robust)
                - "fast": Quick placement for duels (no zoom, just click)
        """
        print(f"   🔧 click_on_map_precise called: method={method}, use_zoom={use_zoom}")
        
        # Only use fast mode if explicitly requested OR if zoom is disabled AND method is not panzoom/doubleclick
        use_fast_mode = method == "fast" or (not use_zoom and method not in ["panzoom", "doubleclick"])
        
        if use_fast_mode:
            # Fast mode: direct click without zoom - good for duels where speed matters
            print(f"   ⚡ Using FAST placement method for ({lat:.4f}, {lng:.4f})")
            self.expand_minimap()
            self.reset_minimap_zoom()  # Ensure we're at world view
            sleep(0.2)
            
            # Calculate target position
            target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
            cal = self.MAP_CALIBRATION
            adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
            adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
            adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
            adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
            
            x = self.map_x + int(adjusted_x_ratio * self.map_w)
            y = self.map_y + int(adjusted_y_ratio * self.map_h)
            x, y = self.clamp_to_minimap(x, y)
            
            pyautogui.click(x, y)
            print(f"   🎯 Fast click at ({x}, {y})")
            sleep(0.2)
            return
            
        if method == "doubleclick":
            print(f"   🎯 Using DOUBLE-CLICK placement method")
            self.smart_place_guess(lat, lng)
        elif method == "panzoom":
            print(f"   🎯 Using PANZOOM placement method")
            self.iterative_pan_zoom_place(lat, lng)
        else:
            # Default: scroll zoom method
            print(f"   🎯 Using SCROLL placement method")
            self.scroll_zoom_place_guess(lat, lng)
        
        sleep(0.3)
    
    def calibration_test(self, test_coords: list = None):
        """
        Test the coordinate placement accuracy with known locations.
        
        Run this manually to verify calibration is correct.
        Adjust x_offset, y_offset, x_scale, y_scale in smart_place_guess if needed.
        """
        if test_coords is None:
            # Test with well-known locations
            test_coords = [
                (0, 0, "Gulf of Guinea (0,0)"),
                (48.8566, 2.3522, "Paris"),
                (40.7128, -74.0060, "New York"),
                (-33.8688, 151.2093, "Sydney"),
                (35.6762, 139.6503, "Tokyo"),
                (-22.9068, -43.1729, "Rio de Janeiro"),
            ]
        
        print("\n" + "="*50)
        print("🧪 CALIBRATION TEST")
        print("="*50)
        print("Watch where the cursor moves - it should match the location names")
        print("Press Ctrl+C to stop")
        print("="*50)
        
        for lat, lng, name in test_coords:
            print(f"\n📍 Testing: {name} ({lat:.4f}, {lng:.4f})")
            
            # Expand and reset
            self.expand_minimap()
            self.reset_minimap_zoom()
            
            # Calculate position using class calibration
            target_x_ratio, target_y_ratio = self.lat_lng_to_mercator(lat, lng)
            
            cal = self.MAP_CALIBRATION
            adjusted_x_ratio = cal["x_offset"] + target_x_ratio * cal["x_scale"]
            adjusted_y_ratio = cal["y_offset"] + target_y_ratio * cal["y_scale"]
            adjusted_x_ratio = max(0.05, min(0.95, adjusted_x_ratio))
            adjusted_y_ratio = max(0.05, min(0.95, adjusted_y_ratio))
            
            target_x = self.map_x + int(adjusted_x_ratio * self.map_w)
            target_y = self.map_y + int(adjusted_y_ratio * self.map_h)
            
            print(f"   Calculated pixel: ({target_x}, {target_y})")
            print(f"   Ratios: x={adjusted_x_ratio:.4f}, y={adjusted_y_ratio:.4f}")
            
            # Move cursor to show where we would click
            pyautogui.moveTo(target_x, target_y, duration=0.5)
            sleep(2)  # Pause to observe

    def click_on_map(self, x: int, y: int):
        """Click on the minimap at the specified pixel location."""
        pyautogui.click(x, y)
        sleep(0.3)

    def click_confirm(self):
        """Click the confirm/guess button."""
        pyautogui.click(self.confirm_button)
        sleep(0.5)

    def next_round(self):
        """Advance to the next round."""
        sleep(2)
        
        if self.next_round_button:
            pyautogui.click(self.next_round_button)
        else:
            pyautogui.press("space")
        
        sleep(2)


def play_round(
    bot: GeoBot,
    round_num: int,
    save_screenshots: bool = True,
    tracker: Optional[ResultsTracker] = None,
    auto_advance: bool = True,
    baseline_api_url: Optional[str] = None,
    human_mode: bool = False,
    duel_mode: bool = False,
    placement_method: str = "scroll",
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Play a single round of GeoGuessr.
    
    Args:
        bot: GeoBot instance
        round_num: Round number
        save_screenshots: Whether to save screenshots
        tracker: Results tracker instance
        auto_advance: Whether to automatically advance to next round
        baseline_api_url: Optional baseline API URL for comparison mode
        human_mode: If True, skip automated gameplay (user plays manually)
        duel_mode: If True, skip marker extraction for true location
        placement_method: "scroll", "doubleclick", "panzoom", or "fast"
    
    Returns:
        Tuple of (success, (pred_lat, pred_lng) or None)
    """
    print(f"\n{'='*50}")
    print(f"🎮 ROUND {round_num}")
    print(f"{'='*50}")
    
    pred_lat, pred_lng = None, None
    
    # Wait for panorama to load
    print("⏳ Waiting for panorama to load...")
    sleep(2)

    # Panorama ID extraction disabled - not needed for basic bot operation
    # The true location will be extracted from the results screen marker

    # Take screenshot
    print("📸 Taking screenshot...")
    image = pyautogui.screenshot(region=bot.screen_xywh)
    
    if save_screenshots:
        os.makedirs("screenshots", exist_ok=True)
        image.save(f"screenshots/round_{round_num}.png")
        print(f"   Saved to screenshots/round_{round_num}.png")
    
    # Get prediction from main ML model (port 5000) - used for gameplay
    print("🔮 Getting ML model prediction (main model)...")
    result = bot.predict_location(image)
    
    if result is None:
        print("❌ Failed to get prediction from main model!")
        x = bot.map_x + bot.map_w // 2
        y = bot.map_y + bot.map_h // 2
        print(f"   Using fallback: center of map ({x}, {y})")
        use_precision = False
    else:
        pred_lat, pred_lng = result
        print(f"📍 Predicted (main): {pred_lat:.4f}, {pred_lng:.4f}")
        
        # Record main model prediction if tracker is active
        if tracker:
            tracker.record_prediction(round_num, pred_lat, pred_lng, is_baseline=False)
        
        # Convert to screen coordinates (for fallback)
        x, y = bot.lat_lon_to_screen_coords(pred_lat, pred_lng)
        
        # Clamp to minimap bounds
        x, y = bot.clamp_to_minimap(x, y)
        use_precision = True
    
    # If comparison mode, get prediction from baseline model (port 5002)
    if baseline_api_url:
        print("🔮 Getting ML model prediction (baseline model)...")
        baseline_result = bot.predict_location_from_api(image, baseline_api_url)
        
        if baseline_result is None:
            print("⚠️ Failed to get prediction from baseline model!")
        else:
            baseline_lat, baseline_lng = baseline_result
            print(f"📍 Predicted (baseline): {baseline_lat:.4f}, {baseline_lng:.4f}")
            
            # Record baseline prediction if tracker is active
            if tracker:
                tracker.record_prediction(round_num, baseline_lat, baseline_lng, is_baseline=True)
    
    # Skip gameplay automation if human mode
    if human_mode:
        print("👤 HUMAN MODE: Skipping automated gameplay")
        print("   Model predictions recorded. Please play manually.")
        print("   After you complete the round, press ENTER here to continue...")
        input()
        
        # Wait for results screen to fully load
        sleep(3)
        
        # Capture round result (true location + score) if tracker is active
        # This will also prompt for human input in human mode
        if tracker:
            tracker.capture_round_result(round_num, bot.screen_regions, human_mode=True)
        
        print(f"✅ Round {round_num} complete!")
        return True, (pred_lat, pred_lng) if pred_lat is not None else None
    
    # Normal automated gameplay
    # Click on predicted location with precision zoom
    # (smart_place_guess handles minimap expansion internally)
    # Only use main model's prediction for gameplay
    print("🖱️  Placing guess on map (using main model prediction)...")
    if use_precision and pred_lat is not None:
        # Determine placement method and zoom setting
        # In duel mode, default to "fast" unless placement_method is explicitly set
        if duel_mode and placement_method == "scroll":
            method = "fast"
            use_zoom_for_placement = False
        else:
            method = placement_method
            # Enable zoom for panzoom/doubleclick/scroll methods, disable only for fast
            use_zoom_for_placement = method != "fast"
        
        print(f"   📍 Placement method: {method} (duel_mode={duel_mode}, use_zoom={use_zoom_for_placement})")
        bot.click_on_map_precise(pred_lat, pred_lng, use_zoom=use_zoom_for_placement, method=method)
    else:
        bot.expand_minimap()
        bot.click_on_map(x, y)
    
    # Click confirm button
    print("✅ Confirming guess...")
    bot.click_confirm()
    
    # Wait for results screen (shorter wait in duel mode)
    if duel_mode:
        sleep(1)  # Duels move fast, don't need long wait
    else:
        sleep(3)
    
    # Capture round result (true location + score) if tracker is active
    # Skip marker extraction in duel mode
    if tracker and not duel_mode:
        tracker.capture_round_result(round_num, bot.screen_regions)
    elif duel_mode:
        print("   ⚡ Duel mode: skipping true location extraction")
    
    # Move to next round (only if auto-advancing)
    if auto_advance:
        print("⏭️  Moving to next round...")
        bot.next_round()
    else:
        print("⏸️  Manual control - waiting for user to advance")
    
    print(f"✅ Round {round_num} complete!")
    return True, (pred_lat, pred_lng) if pred_lat is not None else None


def play_game(
    bot: GeoBot,
    num_rounds: int = 5,
    save_screenshots: bool = True,
    track_results: bool = True,
    results_output_dir: str = "results",
    auto_advance: bool = True,
    baseline_api_url: Optional[str] = None,
    human_mode: bool = False,
    duel_mode: bool = False,
    placement_method: str = "scroll",
):
    """
    Play a full game of GeoGuessr.

    Args:
        bot: GeoBot instance
        num_rounds: Number of rounds to play
        save_screenshots: Whether to save screenshots locally
        track_results: Whether to track true locations and scores
        results_output_dir: Directory to save results CSV
        auto_advance: Whether to automatically advance to next round
        baseline_api_url: Optional baseline API URL for comparison mode
        human_mode: If True, disable automated gameplay (user plays manually)
        duel_mode: If True, skip marker extraction (for competitive play)
        placement_method: "scroll", "doubleclick", "panzoom", or "fast"
    """
    print("\n" + "="*60)
    print("🎮 GEOGUESSR ML BOT - STARTING GAME")
    print("="*60)
    print(f"   Rounds to play: {num_rounds}")
    print(f"   API endpoint: {bot.api_url}")
    if baseline_api_url:
        print(f"   Baseline API: {baseline_api_url} (comparison mode)")
    if human_mode:
        print(f"   Mode: HUMAN (manual gameplay)")
    if duel_mode:
        print(f"   Mode: DUEL (no marker extraction, fast placement)")
    print(f"   Placement method: {placement_method}")
    print(f"   Track results: {track_results}")
    print(f"   Auto advance: {auto_advance}")
    print("="*60)
    
    tracker = None
    if track_results:
        tracker = ResultsTracker(output_dir=results_output_dir)
        tracker.connect()  # Always succeeds for PyAutoGUI tracker
    
    for round_num in range(1, num_rounds + 1):
        success, prediction = play_round(
            bot, round_num, save_screenshots, tracker, auto_advance, 
            baseline_api_url, human_mode, duel_mode, placement_method
        )
        if not success:
            print(f"⚠️ Round {round_num} had issues, continuing...")

        # Wait for user input if not auto-advancing
        if not auto_advance and round_num < num_rounds:
            print(f"\n⏸️  Round {round_num} complete. Press ENTER to continue to round {round_num + 1}...")
            input()
    
    # Print summary
    if tracker:
        tracker.print_summary()
        tracker.close()
    
    print("\n" + "="*60)
    print("🎉 GAME COMPLETE!")
    print("="*60)
