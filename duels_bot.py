#!/usr/bin/env python3
"""
GeoGuessr Duels Bot - Plays as opponent in duels using API.

This bot:
1. Connects to a Chrome instance running the bot's GeoGuessr account
2. Extracts panorama ID from the current round
3. Downloads panorama image from Google Street View
4. Sends image to ML model API for prediction
5. Submits guess via GeoGuessr API with UI refresh

Usage:
    # Start Chrome with remote debugging for the bot account
    google-chrome --remote-debugging-port=9223 --user-data-dir=/tmp/bot-profile
    
    # Log into GeoGuessr with bot account in that browser
    # Start a duel and invite yourself (on main account)
    
    # Run the bot
    python duels_bot.py --game-url "https://www.geoguessr.com/duels/xyz123"
"""

import argparse
import base64
import io
import json
import math
import re
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable

import requests
from PIL import Image

# Try to import selenium - needed for panorama extraction
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium not available - panorama extraction will be limited")

from geoguessr_api import GeoGuessrAPI, GuessResult


@dataclass
class RoundState:
    """State for a single round in a duel."""
    round_number: int
    pano_id: Optional[str] = None
    heading: float = 0.0
    pitch: float = 0.0
    zoom: float = 0.0
    lat: Optional[float] = None
    lng: Optional[float] = None
    predicted_lat: Optional[float] = None
    predicted_lng: Optional[float] = None
    score: Optional[int] = None  # Points (live-challenge) or 0 for duels
    damage: Optional[int] = None  # Damage dealt in duels (negative = we took damage)
    distance_meters: Optional[float] = None
    guess_submitted: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class DuelState:
    """State for an entire duel game."""
    game_id: str
    game_type: str = "duels"  # "duels" or "battle-royale"
    current_round: int = 1
    total_rounds: int = 5
    rounds: List[RoundState] = field(default_factory=list)
    bot_total_score: int = 0
    opponent_total_score: int = 0
    game_finished: bool = False


class PanoramaDownloader:
    """Downloads panorama images from Google Street View using pano IDs."""
    
    # Google Street View tile servers (try multiple formats)
    # New format used by GeoGuessr
    TILE_URL_NEW = "https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=apiv3&panoid={pano_id}&output=tile&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
    # Old CBK format (fallback)
    TILE_URL_OLD = "https://cbk0.google.com/cbk?output=tile&panoid={pano_id}&zoom={zoom}&x={x}&y={y}"
    
    # Tile dimensions at different zoom levels
    ZOOM_DIMENSIONS = {
        0: (1, 1),      # 512x512
        1: (2, 1),      # 1024x512
        2: (4, 2),      # 2048x1024
        3: (8, 4),      # 4096x2048
        4: (16, 8),     # 8192x4096
        5: (32, 16),    # 16384x8192
    }
    
    TILE_SIZE = 512
    
    def __init__(self, timeout: int = 10):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        self.timeout = timeout
        self.use_new_api = True  # Start with new API
    
    def download_tile(self, pano_id: str, zoom: int, x: int, y: int) -> Optional[Image.Image]:
        """Download a single panorama tile."""
        # Try new API first
        if self.use_new_api:
            url = self.TILE_URL_NEW.format(pano_id=pano_id, zoom=zoom, x=x, y=y)
        else:
            url = self.TILE_URL_OLD.format(pano_id=pano_id, zoom=zoom, x=x, y=y)
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            elif response.status_code == 404 and self.use_new_api:
                # Try old API as fallback
                self.use_new_api = False
                url = self.TILE_URL_OLD.format(pano_id=pano_id, zoom=zoom, x=x, y=y)
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return Image.open(io.BytesIO(response.content))
            return None
        except Exception as e:
            print(f"   ⚠️ Failed to download tile ({x},{y}): {e}")
            return None
    
    def download_panorama(
        self, 
        pano_id: str, 
        zoom: int = 2,
        heading: float = 0.0,
        fov: float = 90.0,
        output_size: Tuple[int, int] = (1024, 512)
    ) -> Optional[Image.Image]:
        """
        Download a panorama and optionally extract a view at specific heading.
        
        Args:
            pano_id: Google Street View panorama ID
            zoom: Tile zoom level (0-5, higher = more detail)
            heading: View direction in degrees (0 = north, 90 = east)
            fov: Field of view in degrees
            output_size: Output image size (width, height)
        
        Returns:
            PIL Image or None if download failed
        """
        if pano_id is None:
            print("   ❌ No panorama ID provided")
            return None
        
        print(f"   📥 Downloading panorama {pano_id[:20]}... (zoom={zoom})")
        
        # Get tile grid dimensions for this zoom level
        if zoom not in self.ZOOM_DIMENSIONS:
            zoom = 2  # Default to reasonable zoom
        
        cols, rows = self.ZOOM_DIMENSIONS[zoom]
        
        # Download all tiles
        tiles = {}
        for y in range(rows):
            for x in range(cols):
                tile = self.download_tile(pano_id, zoom, x, y)
                if tile:
                    tiles[(x, y)] = tile
        
        if not tiles:
            print("   ❌ Failed to download any tiles")
            return None
        
        # Stitch tiles into full panorama
        pano_width = cols * self.TILE_SIZE
        pano_height = rows * self.TILE_SIZE
        panorama = Image.new('RGB', (pano_width, pano_height))
        
        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))
        
        print(f"   ✅ Downloaded panorama: {pano_width}x{pano_height}")
        
        # Extract view at specific heading if requested
        if heading != 0.0 or fov != 360.0:
            panorama = self.extract_view(panorama, heading, fov, output_size)
        else:
            # Just resize to output size
            panorama = panorama.resize(output_size, Image.Resampling.LANCZOS)
        
        return panorama
    
    def extract_view(
        self, 
        panorama: Image.Image, 
        heading: float, 
        fov: float,
        output_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Extract a view from a 360° panorama at a specific heading.
        
        Args:
            panorama: Full 360° panorama image
            heading: Direction to look (0-360 degrees)
            fov: Field of view in degrees
            output_size: Output image size
        
        Returns:
            Extracted view as PIL Image
        """
        pano_width, pano_height = panorama.size
        
        # Calculate crop region based on heading and FOV
        # Heading 0 = center of image, heading increases to the right
        center_x = int((heading / 360.0) * pano_width) % pano_width
        crop_width = int((fov / 360.0) * pano_width)
        
        # Calculate left and right bounds (handling wrap-around)
        left = center_x - crop_width // 2
        right = center_x + crop_width // 2
        
        if left < 0:
            # Wrap around left edge
            left_part = panorama.crop((pano_width + left, 0, pano_width, pano_height))
            right_part = panorama.crop((0, 0, right, pano_height))
            view = Image.new('RGB', (crop_width, pano_height))
            view.paste(left_part, (0, 0))
            view.paste(right_part, (-left, 0))
        elif right > pano_width:
            # Wrap around right edge
            left_part = panorama.crop((left, 0, pano_width, pano_height))
            right_part = panorama.crop((0, 0, right - pano_width, pano_height))
            view = Image.new('RGB', (crop_width, pano_height))
            view.paste(left_part, (0, 0))
            view.paste(right_part, (pano_width - left, 0))
        else:
            view = panorama.crop((left, 0, right, pano_height))
        
        # Resize to output dimensions
        return view.resize(output_size, Image.Resampling.LANCZOS)
    
    def download_multiple_views(
        self,
        pano_id: str,
        headings: List[float] = [0, 90, 180, 270],
        zoom: int = 2,
        fov: float = 90.0,
        output_size: Tuple[int, int] = (640, 480)
    ) -> List[Image.Image]:
        """
        Download a panorama and extract views at multiple headings.
        
        Useful for models that expect multiple viewing angles.
        """
        # Download full panorama once
        full_pano = self._download_full_panorama(pano_id, zoom)
        if full_pano is None:
            return []
        
        views = []
        for heading in headings:
            view = self.extract_view(full_pano, heading, fov, output_size)
            views.append(view)
        
        return views
    
    def _download_full_panorama(self, pano_id: str, zoom: int = 2) -> Optional[Image.Image]:
        """Download full 360° panorama without cropping."""
        if zoom not in self.ZOOM_DIMENSIONS:
            zoom = 2
        
        cols, rows = self.ZOOM_DIMENSIONS[zoom]
        tiles = {}
        
        for y in range(rows):
            for x in range(cols):
                tile = self.download_tile(pano_id, zoom, x, y)
                if tile:
                    tiles[(x, y)] = tile
        
        if not tiles:
            return None
        
        pano_width = cols * self.TILE_SIZE
        pano_height = rows * self.TILE_SIZE
        panorama = Image.new('RGB', (pano_width, pano_height))
        
        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * self.TILE_SIZE, y * self.TILE_SIZE))
        
        return panorama


class DuelsBot:
    """
    Bot that plays GeoGuessr duels using the API.
    
    Connects to a separate Chrome instance running the bot's account,
    extracts panorama data, and submits guesses via API.
    """
    
    def __init__(
        self,
        chrome_debug_port: int = 9223,
        ml_api_url: str = "http://127.0.0.1:5000/api/v1/predict",
        cookies_file: Optional[str] = None,
        use_screenshot: bool = True,  # Use screenshot instead of panorama download
        on_status_update: Optional[Callable[[str], None]] = None,
        on_round_start: Optional[Callable[[int, Optional[str]], None]] = None,
        on_prediction: Optional[Callable[[float, float], None]] = None,
        on_guess_result: Optional[Callable[[GuessResult], None]] = None,
        on_round_complete: Optional[Callable[['RoundState'], None]] = None,
        on_game_end: Optional[Callable[['DuelState'], None]] = None,
    ):
        """
        Initialize the Duels Bot.
        
        Args:
            chrome_debug_port: Port for Chrome remote debugging (bot's browser)
            ml_api_url: URL for the ML prediction API
            cookies_file: Path to cookies JSON file (alternative to Selenium)
            use_screenshot: If True (default), use browser screenshot instead of downloading panorama
            on_status_update: Callback for status messages
            on_round_start: Callback when round starts (round_num, pano_id)
            on_prediction: Callback when prediction is made (lat, lng)
            on_guess_result: Callback when guess result received (GuessResult)
            on_round_complete: Callback when round completes (RoundState)
            on_game_end: Callback when game ends (DuelState)
        """
        self.chrome_port = chrome_debug_port
        self.ml_api_url = ml_api_url
        self.cookies_file = cookies_file
        self.use_screenshot = use_screenshot
        
        # Callbacks for UI updates
        self.on_status_update = on_status_update
        self.on_round_start = on_round_start
        self.on_prediction = on_prediction
        self.on_guess_result = on_guess_result
        self.on_round_complete = on_round_complete
        self.on_game_end = on_game_end
        
        # Components
        self.api = GeoGuessrAPI()
        self.panorama_downloader = PanoramaDownloader()
        self.driver: Optional[webdriver.Chrome] = None
        
        # State
        self.current_game: Optional[DuelState] = None
        self.should_stop = threading.Event()
        self._connected = False
        self._user_id: Optional[str] = None  # Cached user ID
        
        # Load cookies if provided
        if cookies_file and Path(cookies_file).exists():
            self.api.load_cookies_from_file(cookies_file)
            self._log(f"✅ Loaded cookies from {cookies_file}")
    
    def _log(self, message: str):
        """Log a message and optionally call status callback."""
        print(message)
        if self.on_status_update:
            try:
                self.on_status_update(message)
            except:
                pass
    
    def connect_to_chrome(self) -> bool:
        """Connect to Chrome browser with remote debugging."""
        if not SELENIUM_AVAILABLE:
            self._log("❌ Selenium not available")
            return False
        
        try:
            options = Options()
            options.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.chrome_port}")
            
            self.driver = webdriver.Chrome(options=options)
            self._log(f"✅ Connected to Chrome on port {self.chrome_port}")
            self._log(f"   Current URL: {self.driver.current_url}")
            
            # Load cookies from browser
            self.api.load_cookies_from_selenium(self.driver)
            self._connected = True
            
            return True
        except Exception as e:
            self._log(f"❌ Failed to connect to Chrome: {e}")
            self._log(f"   Make sure Chrome is running with: --remote-debugging-port={self.chrome_port}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Chrome."""
        return self._connected and self.driver is not None
    
    def get_user_id(self) -> Optional[str]:
        """Get the current user's ID (cached after first call)."""
        if self._user_id:
            return self._user_id
        
        try:
            user_info = self.api.get_user_info()
            if user_info:
                self._user_id = user_info.get("user", {}).get("id")
                if self._user_id:
                    self._log(f"   👤 User ID: {self._user_id[:8]}...")
                return self._user_id
        except Exception as e:
            self._log(f"   ⚠️ Failed to get user ID: {e}")
        return None
    
    def get_game_id_from_url(self, url: Optional[str] = None) -> Optional[str]:
        """Extract game ID from URL."""
        if url is None and self.driver:
            url = self.driver.current_url
        
        if not url:
            return None
        
        patterns = [
            r'/live-challenge/([A-Za-z0-9\-]+)',  # UUID format for live challenges
            r'/duels/([A-Za-z0-9\-]+)',
            r'/battle-royale/([A-Za-z0-9\-]+)',
            r'/game/([A-Za-z0-9]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    def get_game_type_from_url(self, url: Optional[str] = None) -> str:
        """Determine game type from URL."""
        if url is None and self.driver:
            url = self.driver.current_url
        
        if url:
            if '/live-challenge/' in url:
                return "live-challenge"
            elif '/duels/' in url:
                return "duels"
            elif '/battle-royale/' in url:
                return "battle-royale"
        
        return "classic"
    
    def extract_pano_id_from_browser(self) -> Optional[str]:
        """Extract panorama ID from the browser using multiple methods."""
        if not self.driver:
            return None
        
        pano_id = None
        
        # Method 1: Check performance entries for Street View tile URLs (most reliable)
        try:
            js_code = """
            const entries = performance.getEntriesByType('resource');
            const panoIds = new Set();
            
            for (const entry of entries) {
                // Check for streetviewpixels URLs (new format)
                let match = entry.name.match(/streetviewpixels.*panoid=([A-Za-z0-9_-]+)/);
                if (match) {
                    panoIds.add(match[1]);
                }
                // Check for cbk URLs (old format)
                if (!match) {
                    match = entry.name.match(/cbk.*panoid=([A-Za-z0-9_-]+)/);
                    if (match) {
                        panoIds.add(match[1]);
                    }
                }
            }
            
            // Return the most recent one (last in array)
            const ids = Array.from(panoIds);
            return ids.length > 0 ? ids[ids.length - 1] : null;
            """
            pano_id = self.driver.execute_script(js_code)
            if pano_id:
                self._log(f"   📍 Found pano ID from network: {pano_id[:20]}...")
                return pano_id
        except Exception as e:
            self._log(f"   ⚠️ Network extraction failed: {e}")
        
        # Method 2: Check for Google Maps iframe URL
        try:
            iframes = self.driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                src = iframe.get_attribute("src") or ""
                if "maps.google.com" in src or "streetview" in src.lower():
                    match = re.search(r'pano[=:]([A-Za-z0-9_-]+)', src)
                    if match:
                        pano_id = match.group(1)
                        self._log(f"   📍 Found pano ID in iframe: {pano_id[:20]}...")
                        return pano_id
        except Exception as e:
            pass
        
        # Method 3: Execute JavaScript to find pano ID in page source
        try:
            js_code = """
            // Check page source for panoId patterns
            const html = document.documentElement.innerHTML;
            const patterns = [
                /"panoId":"([A-Za-z0-9_-]+)"/,
                /"pano":"([A-Za-z0-9_-]+)"/,
                /panoid=([A-Za-z0-9_-]+)/,
            ];
            for (const pattern of patterns) {
                const match = html.match(pattern);
                if (match) {
                    return match[1];
                }
            }
            return null;
            """
            pano_id = self.driver.execute_script(js_code)
            if pano_id:
                self._log(f"   📍 Found pano ID in HTML: {pano_id[:20]}...")
                return pano_id
        except Exception as e:
            pass
        
        # Method 4: Parse page source directly with Python
        try:
            html = self.driver.page_source
            patterns = [
                r'"panoId":"([A-Za-z0-9_-]+)"',
                r'"pano":"([A-Za-z0-9_-]+)"',
                r'panoid=([A-Za-z0-9_-]+)',
                r'"imageDataId":"([A-Za-z0-9_-]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    pano_id = match.group(1)
                    self._log(f"   📍 Found pano ID in source: {pano_id[:20]}...")
                    return pano_id
        except Exception as e:
            pass
        
        return pano_id
    
    def get_round_data_from_api(self, game_id: str, game_type: str) -> Optional[Dict[str, Any]]:
        """Get current round data from GeoGuessr API."""
        try:
            if game_type == "duels":
                url = f"https://game-server.geoguessr.com/api/duels/{game_id}"
            elif game_type == "live-challenge":
                url = f"https://game-server.geoguessr.com/api/live-challenge/{game_id}"
            elif game_type == "battle-royale":
                url = f"https://game-server.geoguessr.com/api/battle-royale/{game_id}"
            else:
                url = f"https://www.geoguessr.com/api/v3/games/{game_id}"
            
            response = self.api.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            self._log(f"   ⚠️ Failed to get round data: {e}")
            return None
    
    def get_pano_id_from_api(self, game_id: str, game_type: str, round_number: int) -> Optional[str]:
        """Get panorama ID from API for a specific round."""
        game_data = self.get_round_data_from_api(game_id, game_type)
        if not game_data:
            return None
        
        # Try to find pano ID in the game data
        # Structure varies by game type
        try:
            if game_type in ("duels", "live-challenge"):
                rounds = game_data.get("rounds", [])
                if round_number <= len(rounds):
                    round_data = rounds[round_number - 1]
                    return round_data.get("panoId") or round_data.get("panoramaId")
            elif game_type == "classic":
                rounds = game_data.get("rounds", [])
                if round_number <= len(rounds):
                    round_data = rounds[round_number - 1]
                    return round_data.get("panoId")
        except Exception as e:
            pass
        
        return None
    
    def take_panorama_screenshot(self) -> Optional[Image.Image]:
        """
        Take a screenshot of just the panorama area (excluding UI elements).
        This is faster and more compatible than downloading panorama tiles.
        """
        if not self.driver:
            return None
        
        try:
            # First try to find the panorama canvas/container element
            js_code = """
            // Find the panorama element
            const selectors = [
                '[class*="panorama"]',
                '[class*="game-panorama"]', 
                '[class*="game-canvas"]',
                'canvas'
            ];
            
            for (const selector of selectors) {
                const el = document.querySelector(selector);
                if (el && el.offsetWidth > 500) {
                    const rect = el.getBoundingClientRect();
                    return {
                        x: Math.round(rect.x),
                        y: Math.round(rect.y),
                        width: Math.round(rect.width),
                        height: Math.round(rect.height)
                    };
                }
            }
            
            // Fallback: return full viewport minus some margins for UI
            return {
                x: 0,
                y: 0,
                width: window.innerWidth,
                height: Math.round(window.innerHeight * 0.85)  // Exclude bottom UI
            };
            """
            
            bounds = self.driver.execute_script(js_code)
            
            # Take full screenshot
            png = self.driver.get_screenshot_as_png()
            full_img = Image.open(io.BytesIO(png))
            
            # Crop to panorama area if bounds found
            if bounds and bounds.get('width', 0) > 100:
                x = max(0, bounds['x'])
                y = max(0, bounds['y'])
                w = bounds['width']
                h = bounds['height']
                
                # Make sure we don't exceed image bounds
                x2 = min(x + w, full_img.width)
                y2 = min(y + h, full_img.height)
                
                cropped = full_img.crop((x, y, x2, y2))
                self._log(f"   ✅ Screenshot captured: {cropped.size}")
                return cropped
            else:
                # Return full screenshot
                self._log(f"   ✅ Full screenshot captured: {full_img.size}")
                return full_img
                
        except Exception as e:
            self._log(f"   ⚠️ Screenshot failed: {e}")
            # Fallback to simple screenshot
            try:
                png = self.driver.get_screenshot_as_png()
                img = Image.open(io.BytesIO(png))
                self._log(f"   ✅ Fallback screenshot: {img.size}")
                return img
            except:
                return None
    
    def send_to_ml_api(self, image: Image.Image) -> Optional[Tuple[float, float]]:
        """Send image to ML API and get prediction."""
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        payload = {"image": f"data:image/png;base64,{img_b64}"}
        
        try:
            self._log(f"   🔮 Sending to ML API...")
            response = requests.post(
                self.ml_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                lat = result["results"]["lat"]
                lng = result["results"]["lng"]
                self._log(f"   📍 Prediction: ({lat:.4f}, {lng:.4f})")
                return lat, lng
            else:
                self._log(f"   ❌ ML API error: {response.status_code}")
                return None
        except Exception as e:
            self._log(f"   ❌ ML API request failed: {e}")
            return None
    
    def submit_guess(
        self, 
        game_id: str, 
        lat: float, 
        lng: float, 
        round_number: int,
        game_type: str = "duels"
    ) -> GuessResult:
        """Submit a guess via the GeoGuessr API."""
        return self.api.submit_guess(
            game_id=game_id,
            lat=lat,
            lng=lng,
            round_number=round_number,
            game_type=game_type
        )
    
    def trigger_ui_refresh(self):
        """Trigger a UI refresh in the browser after submitting a guess."""
        if not self.driver:
            return
        
        try:
            # Execute JavaScript to dispatch events and trigger UI update
            self.driver.execute_script("""
                // Dispatch focus event
                window.dispatchEvent(new Event('focus'));
                
                // Dispatch visibility change
                document.dispatchEvent(new Event('visibilitychange'));
                
                // Try to find and click elements that might trigger updates
                const selectors = [
                    '[class*="guess-map__guess-button"]',
                    '[class*="game_guess"]',
                    '[data-qa="perform-guess"]',
                    'button[class*="guess"]'
                ];
                
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el) {
                        // Simulate hover to trigger any lazy updates
                        el.dispatchEvent(new MouseEvent('mouseenter', {bubbles: true}));
                        break;
                    }
                }
                
                // Force React to re-render by triggering a resize
                window.dispatchEvent(new Event('resize'));
            """)
            self._log("   🔄 Triggered UI refresh")
        except Exception as e:
            self._log(f"   ⚠️ Could not trigger UI refresh: {e}")
    
    def click_guess_button(self):
        """Click the guess button to submit via UI (triggers proper animations)."""
        if not self.driver:
            return False
        
        try:
            selectors = [
                '[class*="guess-map__guess-button"]',
                '[class*="game_guess"]',
                '[data-qa="perform-guess"]',
                'button[class*="guess"]'
            ]
            
            for selector in selectors:
                try:
                    btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if btn and btn.is_displayed():
                        btn.click()
                        self._log("   🖱️ Clicked guess button")
                        return True
                except:
                    continue
            
            return False
        except Exception as e:
            self._log(f"   ⚠️ Could not click guess button: {e}")
            return False
    
    def place_marker_on_map(self, lat: float, lng: float):
        """Place a marker on the guess map via JavaScript."""
        if not self.driver:
            return
        
        try:
            # This JavaScript simulates clicking on the guess map at the target location
            # It works with Leaflet-based maps used by GeoGuessr
            js_code = f"""
            (function() {{
                const lat = {lat};
                const lng = {lng};
                
                // Find the Leaflet map container
                const mapContainer = document.querySelector('.guess-map') || 
                                    document.querySelector('[class*="guess-map"]') ||
                                    document.querySelector('.leaflet-container');
                
                if (!mapContainer) {{
                    console.log('Map container not found');
                    return false;
                }}
                
                // Try to access the Leaflet map instance
                let map = null;
                
                // Method 1: Check for _leaflet_map property
                if (mapContainer._leaflet_map) {{
                    map = mapContainer._leaflet_map;
                }}
                
                // Method 2: Search in L.maps registry
                if (!map && window.L) {{
                    for (const id in L._maps || {{}}) {{
                        map = L._maps[id];
                        break;
                    }}
                }}
                
                if (map) {{
                    // Set view to target location
                    map.setView([lat, lng], 8);
                    
                    // Simulate a click at the center of the map
                    const center = map.getCenter();
                    const point = map.latLngToContainerPoint(center);
                    
                    const clickEvent = new MouseEvent('click', {{
                        bubbles: true,
                        cancelable: true,
                        view: window,
                        clientX: mapContainer.getBoundingClientRect().left + point.x,
                        clientY: mapContainer.getBoundingClientRect().top + point.y
                    }});
                    
                    mapContainer.dispatchEvent(clickEvent);
                    console.log('Placed marker via Leaflet');
                    return true;
                }}
                
                // Fallback: Just click in center of map element
                const rect = mapContainer.getBoundingClientRect();
                const event = new MouseEvent('click', {{
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: rect.left + rect.width / 2,
                    clientY: rect.top + rect.height / 2
                }});
                mapContainer.dispatchEvent(event);
                console.log('Placed marker via click');
                return true;
            }})();
            """
            self.driver.execute_script(js_code)
            self._log(f"   📍 Placed marker at ({lat:.4f}, {lng:.4f})")
        except Exception as e:
            self._log(f"   ⚠️ Could not place marker: {e}")
    
    def wait_for_round_ready(self, timeout: int = 30) -> bool:
        """Wait for a round to be ready (guess button visible)."""
        if not self.driver:
            return False
        
        start = time.time()
        while time.time() - start < timeout:
            if self.should_stop.is_set():
                return False
            
            try:
                selectors = [
                    '[class*="guess-map__guess-button"]',
                    '[class*="game_guess"]',
                    '[data-qa="perform-guess"]'
                ]
                for selector in selectors:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return True
            except:
                pass
            
            time.sleep(0.5)
        
        return False
    
    def wait_for_round_end(self, timeout: int = 60) -> bool:
        """Wait for current round to end (guess button disappears or result shown)."""
        if not self.driver:
            return True
        
        start = time.time()
        while time.time() - start < timeout:
            if self.should_stop.is_set():
                return False
            
            try:
                # Check if result screen is showing
                result_selectors = [
                    '[class*="result-layout"]',
                    '[class*="round-result"]',
                    '[class*="game-finished"]'
                ]
                for selector in result_selectors:
                    if self.driver.find_elements(By.CSS_SELECTOR, selector):
                        return True
                
                # Or if guess button disappeared
                guess_btn = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    '[class*="guess-map__guess-button"], [class*="game_guess"]'
                )
                if not guess_btn:
                    return True
            except:
                pass
            
            time.sleep(0.5)
        
        return False
    
    def play_round(self, game_id: str, round_number: int, game_type: str = "duels") -> Optional[RoundState]:
        """Play a single round of a duel."""
        round_state = RoundState(round_number=round_number)
        
        # Wait for round to be ready
        self._log("⏳ Waiting for round to start...")
        if not self.wait_for_round_ready():
            self._log("❌ Timeout waiting for round")
            return None
        
        time.sleep(2)  # Let panorama load
        
        # Try to get pano ID from API first (more reliable)
        self._log("🔍 Getting panorama ID...")
        pano_id = self.get_pano_id_from_api(game_id, game_type, round_number)
        
        # Fallback to browser extraction
        if not pano_id:
            pano_id = self.extract_pano_id_from_browser()
        
        round_state.pano_id = pano_id
        
        if self.on_round_start:
            try:
                self.on_round_start(round_number, pano_id)
            except:
                pass
        
        # Get image for ML prediction
        image = None
        
        if self.use_screenshot:
            # Take screenshot of the game (default, faster and more compatible)
            self._log("📸 Taking screenshot...")
            image = self.take_panorama_screenshot()
        else:
            # Download panorama tiles (optional, higher quality but slower)
            if pano_id:
                self._log("📥 Downloading panorama...")
                image = self.panorama_downloader.download_panorama(
                    pano_id, 
                    zoom=2,
                    heading=0,
                    fov=90,
                    output_size=(1024, 512)
                )
            
            if image is None:
                self._log("⚠️ No panorama - taking screenshot instead...")
                image = self.take_panorama_screenshot()
        
        if image is None:
            self._log("❌ Failed to get any image")
            return round_state
        
        # Save panorama for debugging
        debug_dir = Path("debug_panoramas")
        debug_dir.mkdir(exist_ok=True)
        image.save(debug_dir / f"round_{round_number}.png")
        
        # Get ML prediction
        self._log("🔮 Getting ML prediction...")
        prediction = self.send_to_ml_api(image)
        
        if prediction is None:
            prediction = (0.0, 0.0)
            self._log("⚠️ Using fallback prediction (0, 0)")
        
        pred_lat, pred_lng = prediction
        round_state.predicted_lat = pred_lat
        round_state.predicted_lng = pred_lng
        
        if self.on_prediction:
            try:
                self.on_prediction(pred_lat, pred_lng)
            except:
                pass
        
        # Place marker on the map (for visual feedback)
        self._log(f"📍 Placing marker at ({pred_lat:.4f}, {pred_lng:.4f})...")
        self.place_marker_on_map(pred_lat, pred_lng)
        time.sleep(0.5)
        
        # Submit guess via API
        self._log(f"📤 Submitting guess...")
        result = self.submit_guess(game_id, pred_lat, pred_lng, round_number, game_type)
        
        round_state.guess_submitted = result.success
        if result.success:
            self._log(f"✅ Guess submitted!")
            # Store results from immediate response if available
            if result.score is not None:
                round_state.score = result.score
            if result.distance_meters is not None:
                round_state.distance_meters = result.distance_meters
            if result.true_lat is not None:
                round_state.lat = result.true_lat
            if result.true_lng is not None:
                round_state.lng = result.true_lng
        else:
            self._log(f"❌ API guess failed: {result.error}")
            # Try clicking the UI button as fallback
            self._log("   🔄 Trying UI button fallback...")
            if self.click_guess_button():
                round_state.guess_submitted = True
                self._log("   ✅ UI fallback succeeded")
        
        if self.on_guess_result:
            try:
                self.on_guess_result(result)
            except:
                pass
        
        # Trigger UI refresh to show the result
        self.trigger_ui_refresh()
        
        # Wait for round to end (all players guess)
        self._log("⏳ Waiting for round to end...")
        self.wait_for_round_end()
        
        # Poll API until round results are available (opponent has guessed)
        self._log("📊 Fetching round results...")
        round_results = None
        max_wait_time = 60  # Maximum seconds to wait for results
        poll_interval = 2  # Seconds between polls
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if self.should_stop.is_set():
                break
                
            round_results = self.get_round_results(game_id, game_type, round_number)
            
            # Check if we got valid results
            # For duels: both score AND damage should be available when round truly ends
            # (having just distance means our guess exists but round hasn't ended yet)
            if round_results:
                has_score = round_results.get("score") is not None
                has_damage = round_results.get("damage") is not None
                
                # For duels, we need BOTH score and damage (both come from roundResults)
                # If we only have distance, the round hasn't ended yet
                if game_type in ("duels", "team-duels"):
                    if has_score and has_damage:
                        self._log(f"   ✓ Results received after {time.time() - start_time:.1f}s")
                        break
                else:
                    # For other modes, score is sufficient
                    if has_score:
                        self._log(f"   ✓ Results received after {time.time() - start_time:.1f}s")
                        break
            
            # Wait before next poll
            time.sleep(poll_interval)
        
        if round_results:
            if round_state.score is None:
                round_state.score = round_results.get("score")
            if round_state.distance_meters is None:
                round_state.distance_meters = round_results.get("distance_meters")
            if round_state.lat is None:
                round_state.lat = round_results.get("true_lat")
            if round_state.lng is None:
                round_state.lng = round_results.get("true_lng")
            # Get damage for duels
            round_state.damage = round_results.get("damage")
        
        # Log the results based on game type
        if game_type in ("duels", "team-duels"):
            # For duels, show damage instead of score
            if round_state.damage is not None:
                if round_state.damage < 0:
                    self._log(f"   ❌ Damage taken: {abs(round_state.damage)} HP")
                elif round_state.damage > 0:
                    self._log(f"   ✅ Damage dealt: {round_state.damage} HP")
                else:
                    self._log(f"   ➖ Tie round (0 damage)")
            if round_state.score is not None:
                self._log(f"   🎯 Score: {round_state.score} pts")
        else:
            # For other modes, show score
            if round_state.score is not None:
                self._log(f"   ✅ Score: {round_state.score} points")
        
        if round_state.distance_meters is not None:
            self._log(f"   📏 Distance: {round_state.distance_meters:.0f} meters")
        if round_state.lat and round_state.lng:
            self._log(f"   📍 True location: ({round_state.lat:.4f}, {round_state.lng:.4f})")
        
        # Invoke round complete callback
        if self.on_round_complete:
            try:
                self.on_round_complete(round_state)
            except Exception as e:
                self._log(f"   ⚠️ Callback error: {e}")
        
        return round_state
    
    def get_round_results(self, game_id: str, game_type: str, round_number: int) -> Optional[Dict[str, Any]]:
        """
        Fetch round results from API after round ends.
        Returns dict with score, damage, distance_meters, true_lat, true_lng, predicted_lat, predicted_lng.
        
        For duels/team-duels: 
            - score = our score (points)
            - damage = health lost (negative = we took damage, 0 = we won the round)
        For live-challenge: score is points earned
        """
        try:
            # Get game data
            game_data = self.get_round_data_from_api(game_id, game_type)
            if not game_data:
                return None
            
            # Get true location from rounds array
            true_lat = None
            true_lng = None
            rounds = game_data.get("rounds", [])
            if rounds and round_number <= len(rounds):
                round_data = rounds[round_number - 1]
                if isinstance(round_data, dict):
                    # Try direct lat/lng first
                    true_lat = round_data.get("lat")
                    true_lng = round_data.get("lng")
                    
                    # Try panorama nested object (duels format)
                    if true_lat is None:
                        panorama = round_data.get("panorama")
                        if isinstance(panorama, dict):
                            true_lat = panorama.get("lat")
                            true_lng = panorama.get("lng")
                    
                    # Try answer.coordinateAnswerPayload.coordinate (live-challenge format)
                    if true_lat is None:
                        answer = round_data.get("answer", {})
                        coord_payload = answer.get("coordinateAnswerPayload", {})
                        coord = coord_payload.get("coordinate", {})
                        if coord:
                            true_lat = coord.get("lat")
                            true_lng = coord.get("lng")
                    
                    # Try question.panoramaQuestionPayload.panorama (another live-challenge path)
                    if true_lat is None:
                        question = round_data.get("question", {})
                        pano_payload = question.get("panoramaQuestionPayload", {})
                        pano = pano_payload.get("panorama", {})
                        if pano:
                            true_lat = pano.get("lat")
                            true_lng = pano.get("lng")
            
            # Initialize results
            score = None
            damage = None
            distance_meters = None
            predicted_lat = None
            predicted_lng = None
            
            # For duels/team-duels, get results from teams.roundResults
            if game_type in ("duels", "team-duels"):
                our_user_id = self.get_user_id()
                teams = game_data.get("teams", [])
                
                if not our_user_id:
                    self._log(f"   ⚠️ Could not get user ID - cannot match team")
                else:
                    found_our_team = False
                    for team in teams:
                        # Check if this is our team
                        players = team.get("players", [])
                        is_our_team = any(p.get("playerId") == our_user_id for p in players)
                        
                        if is_our_team:
                            found_our_team = True
                            round_results_list = team.get("roundResults", [])
                            
                            # Find results for this round
                            for rr in round_results_list:
                                if rr.get("roundNumber") == round_number:
                                    # Our score (points) - only set if present
                                    score = rr.get("score")
                                    
                                    # Damage taken = healthAfter - healthBefore
                                    # Only calculate if health values are actually present (not just defaulting to 0)
                                    health_before = rr.get("healthBefore")
                                    health_after = rr.get("healthAfter")
                                    if health_before is not None and health_after is not None:
                                        damage = health_after - health_before  # Negative = took damage
                                    
                                    # Get our guess details from bestGuess
                                    best_guess = rr.get("bestGuess")
                                    if best_guess:
                                        distance_meters = best_guess.get("distance")
                                        predicted_lat = best_guess.get("lat")
                                        predicted_lng = best_guess.get("lng")
                                    
                                    break
                            
                            # If not found in roundResults, try player guesses
                            if score is None:
                                for player in players:
                                    if player.get("playerId") == our_user_id:
                                        player_guesses = player.get("guesses", [])
                                        for guess in player_guesses:
                                            if guess.get("roundNumber") == round_number:
                                                score = guess.get("score")
                                                distance_meters = guess.get("distance")
                                                predicted_lat = guess.get("lat")
                                                predicted_lng = guess.get("lng")
                                                break
                                        break
                            
                            break
                    
                    if not found_our_team:
                        self._log(f"   ⚠️ Could not find our team in game data")
            
            # For live-challenge and other modes, use guesses array
            else:
                guesses = game_data.get("guesses", [])
                
                if isinstance(guesses, list):
                    for guess in guesses:
                        if isinstance(guess, dict):
                            guess_round = guess.get("roundNumber") or guess.get("round")
                            if guess_round == round_number:
                                score = guess.get("score")
                                distance_meters = guess.get("distance")
                                break
            
            return {
                "score": score,
                "damage": damage,
                "distance_meters": distance_meters,
                "true_lat": true_lat,
                "true_lng": true_lng,
                "predicted_lat": predicted_lat,
                "predicted_lng": predicted_lng,
            }
            
        except Exception as e:
            self._log(f"   ⚠️ Error fetching results: {e}")
            return None
    
    def is_game_finished(self, game_id: str, game_type: str) -> Tuple[bool, Optional[str]]:
        """
        Check if the game is finished.
        
        Returns:
            Tuple of (is_finished, winner) where winner is 'us', 'opponent', 'draw', or None
        """
        try:
            game_data = self.get_round_data_from_api(game_id, game_type)
            if not game_data:
                return False, None
            
            # Check status field (duels use this)
            status = game_data.get("status", "")
            if status.lower() == "finished":
                # Check result
                result = game_data.get("result", {})
                winning_team_id = result.get("winningTeamId")
                is_draw = result.get("isDraw", False)
                
                if is_draw:
                    return True, "draw"
                
                # Find if we won
                our_user_id = self.get_user_id()
                teams = game_data.get("teams", [])
                
                for team in teams:
                    players = team.get("players", [])
                    is_our_team = any(p.get("playerId") == our_user_id for p in players)
                    
                    if team.get("id") == winning_team_id:
                        return True, "us" if is_our_team else "opponent"
                
                return True, None
            
            # Check state field (alternative)
            state = game_data.get("state", "")
            if state.lower() in ("finished", "ended"):
                return True, None
            
            return False, None
            
        except Exception:
            return False, None
    
    def play_game(self, game_url: Optional[str] = None, num_rounds: int = 5) -> Optional[DuelState]:
        """
        Play a full duel game.
        
        Args:
            game_url: URL of the duel game (optional if already on game page)
            num_rounds: Maximum number of rounds (use large number for duels which can vary)
        
        Returns:
            DuelState with game results
        """
        self.should_stop.clear()
        
        # Connect to browser if not already connected
        if not self.is_connected:
            if not self.connect_to_chrome():
                return None
        
        # Navigate to game URL if provided
        if game_url and self.driver:
            self._log(f"🌐 Navigating to {game_url}")
            self.driver.get(game_url)
            time.sleep(3)
        
        # Get game ID and type
        game_id = self.get_game_id_from_url()
        game_type = self.get_game_type_from_url()
        
        if not game_id:
            self._log("❌ Could not determine game ID from URL")
            return None
        
        # Get current game state from API to determine starting round
        game_data = self.get_round_data_from_api(game_id, game_type)
        starting_round = 1
        if game_data:
            # Get current round from API
            api_round = game_data.get("currentRoundNumber") or game_data.get("round")
            if api_round:
                starting_round = api_round
            
            # Check if game is already finished using new method
            is_finished, winner = self.is_game_finished(game_id, game_type)
            if is_finished:
                self._log("❌ Game is already finished")
                return None
        
        self._log(f"\n{'='*60}")
        self._log(f"🎮 STARTING GAME")
        self._log(f"{'='*60}")
        self._log(f"   Game ID: {game_id}")
        self._log(f"   Game Type: {game_type}")
        self._log(f"   ML API: {self.ml_api_url}")
        self._log(f"   Starting from round: {starting_round}")
        
        # For duels/team-duels, rounds are dynamic - don't show fixed number
        if game_type in ("duels", "team-duels"):
            self._log(f"   Rounds: dynamic (until game ends)")
        else:
            self._log(f"   Max Rounds: {num_rounds}")
        self._log(f"{'='*60}")
        
        # Initialize game state
        self.current_game = DuelState(
            game_id=game_id,
            game_type=game_type,
            total_rounds=num_rounds,
            current_round=starting_round,
        )
        
        # Play rounds with manual confirmation - start from detected round
        round_num = starting_round - 1  # Will be incremented at start of loop
        while round_num < num_rounds:
            round_num += 1
            
            if self.should_stop.is_set():
                self._log("\n⚠️ Bot stopped by user")
                break
            
            # Check if game is finished using new method
            is_finished, winner = self.is_game_finished(game_id, game_type)
            if is_finished:
                self._log(f"\n🏁 Game has ended!")
                if winner == "us":
                    self._log("   🎉 We won!")
                elif winner == "opponent":
                    self._log("   😢 Opponent won")
                elif winner == "draw":
                    self._log("   🤝 It's a draw!")
                self.current_game.game_finished = True
                break
            
            # Check current round from API before each round
            game_data = self.get_round_data_from_api(game_id, game_type)
            if game_data:
                # Update round number from API
                api_round = game_data.get("currentRoundNumber") or game_data.get("round")
                if api_round and api_round > round_num:
                    self._log(f"⏭️ Skipping to round {api_round} (detected from API)")
                    round_num = api_round
            
            # Wait for user confirmation before each round
            self._log(f"\n{'='*50}")
            self._log(f"🎮 ROUND {round_num}")
            self._log(f"{'='*50}")
            self._log("\n⏳ Press ENTER when ready to take screenshot...")
            
            # This allows GUI to set confirmation_callback, or CLI uses input()
            if hasattr(self, 'wait_for_confirmation') and callable(self.wait_for_confirmation):
                self.wait_for_confirmation()
            else:
                input()  # CLI mode - wait for Enter key
            
            if self.should_stop.is_set():
                break
            
            round_state = self.play_round(game_id, round_num, game_type)
            
            if round_state:
                self.current_game.rounds.append(round_state)
                if round_state.score:
                    self.current_game.bot_total_score += round_state.score
            
            # Check if game is finished after round
            is_finished, winner = self.is_game_finished(game_id, game_type)
            if is_finished:
                self._log(f"\n🏁 Game has ended!")
                if winner == "us":
                    self._log("   🎉 We won!")
                elif winner == "opponent":
                    self._log("   😢 Opponent won")
                elif winner == "draw":
                    self._log("   🤝 It's a draw!")
                self.current_game.game_finished = True
                break
        
        # Game finished - show summary based on game type
        self._log(f"\n{'='*60}")
        self._log(f"🏆 GAME COMPLETE!")
        self._log(f"{'='*60}")
        
        if game_type in ("duels", "team-duels"):
            # Calculate total damage for duels
            total_damage = sum(getattr(r, 'damage', 0) or 0 for r in self.current_game.rounds)
            total_score = sum(r.score or 0 for r in self.current_game.rounds)
            avg_dist = sum((r.distance_meters or 0) for r in self.current_game.rounds) / max(1, len(self.current_game.rounds))
            
            if total_damage >= 0:
                self._log(f"   Net Damage: +{total_damage} HP (dealt)")
            else:
                self._log(f"   Net Damage: {total_damage} HP (taken)")
            self._log(f"   Total Score: {total_score} pts")
            self._log(f"   Avg Distance: {avg_dist/1000:.1f} km")
        else:
            self._log(f"   Total Score: {self.current_game.bot_total_score}")
        
        self._log(f"   Rounds Played: {len(self.current_game.rounds)}")
        self._log(f"{'='*60}")
        
        if self.on_game_end:
            try:
                self.on_game_end(self.current_game)
            except:
                pass
        
        return self.current_game
    
    def stop(self):
        """Stop the bot."""
        self.should_stop.set()
        self._log("🛑 Bot stopping...")
    
    def close(self):
        """Clean up resources."""
        self.should_stop.set()
        self._connected = False
        # Note: We don't close the driver as it's connected to an existing browser


def run_duels_bot(
    game_url: str,
    chrome_port: int = 9223,
    ml_api_url: str = "http://127.0.0.1:5000/api/v1/predict",
    cookies_file: Optional[str] = None,
    num_rounds: int = 5,
    use_screenshot: bool = True,
) -> Optional[DuelState]:
    """
    Convenience function to run the duels bot.
    
    Args:
        game_url: URL of the duel game
        chrome_port: Chrome remote debugging port for bot account
        ml_api_url: URL of the ML prediction API
        cookies_file: Path to cookies file
        num_rounds: Expected number of rounds
        use_screenshot: If True, take screenshots instead of downloading panoramas
    
    Returns:
        DuelState with game results
    """
    bot = DuelsBot(
        chrome_debug_port=chrome_port,
        ml_api_url=ml_api_url,
        cookies_file=cookies_file,
        use_screenshot=use_screenshot,
    )
    
    try:
        return bot.play_game(game_url, num_rounds=num_rounds)
    finally:
        bot.close()


def main():
    parser = argparse.ArgumentParser(description="GeoGuessr Duels Bot")
    parser.add_argument("--game-url", "-g", help="URL of the duel game")
    parser.add_argument("--chrome-port", "-p", type=int, default=9223,
                       help="Chrome remote debugging port (default: 9223)")
    parser.add_argument("--ml-api", "-a", default="http://127.0.0.1:5000/api/v1/predict",
                       help="ML API URL")
    parser.add_argument("--cookies", "-c", help="Path to cookies JSON file")
    parser.add_argument("--rounds", "-r", type=int, default=5, help="Number of rounds")
    parser.add_argument("--use-panorama", action="store_true",
                       help="Download panorama tiles instead of taking a screenshot (slower but higher quality)")
    
    args = parser.parse_args()
    
    if not args.game_url:
        print(__doc__)
        print("\nQuick Start:")
        print("=" * 50)
        print(f"1. Start Chrome for bot account:")
        print(f"   google-chrome --remote-debugging-port={args.chrome_port} --user-data-dir=/tmp/bot-profile")
        print()
        print("2. Log into GeoGuessr with bot account in that browser")
        print()
        print("3. Create/join a duel game")
        print()
        print("4. Run this script:")
        print(f"   python duels_bot.py --game-url <URL>")
        return
    
    result = run_duels_bot(
        game_url=args.game_url,
        chrome_port=args.chrome_port,
        ml_api_url=args.ml_api,
        cookies_file=args.cookies,
        num_rounds=args.rounds,
        use_screenshot=not args.use_panorama,
    )
    
    if result:
        print(f"\n📊 Final Score: {result.bot_total_score}")
        print(f"📍 Rounds: {len(result.rounds)}")
        for r in result.rounds:
            dist = f"{r.distance_meters:.0f}m" if r.distance_meters else "N/A"
            print(f"   Round {r.round_number}: {r.score or 0} pts ({dist})")


if __name__ == "__main__":
    main()
