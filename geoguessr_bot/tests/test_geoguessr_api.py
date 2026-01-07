#!/usr/bin/env python3
"""
Test script for GeoGuessr API - submit a guess to a game.

Usage:
    python test_geoguessr_api.py <game_url> <lat> <lng> <round>
    
Example:
    python test_geoguessr_api.py https://www.geoguessr.com/game/X8CBIo4RtjoBkgpC 48.8566 2.3522 1
"""

import sys
import json
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from geoguessr_api import GeoGuessrAPI


def main():
    # Parse arguments
    if len(sys.argv) < 5:
        # Default test values
        game_url = "https://www.geoguessr.com/game/X8CBIo4RtjoBkgpC"
        lat = 48.8566  # Paris
        lng = 2.3522
        round_num = 1
        print(f"Using default test values:")
    else:
        game_url = sys.argv[1]
        lat = float(sys.argv[2])
        lng = float(sys.argv[3])
        round_num = int(sys.argv[4])
    
    print(f"  Game URL: {game_url}")
    print(f"  Coordinates: ({lat}, {lng})")
    print(f"  Round: {round_num}")
    print()
    
    # Initialize API
    api = GeoGuessrAPI()
    
    # Try to load cookies from various sources
    cookies_loaded = False
    
    # Method 1: Check for cookies.json file
    cookies_file = Path(__file__).parent / "cookies.json"
    if cookies_file.exists():
        print(f"📂 Loading cookies from {cookies_file}")
        try:
            api.load_cookies_from_file(str(cookies_file))
            cookies_loaded = True
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Method 2: Try Selenium with Chrome remote debugging
    if not cookies_loaded:
        print("🔌 Trying Selenium connection to Chrome (port 9222)...")
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            options = Options()
            options.add_experimental_option('debuggerAddress', '127.0.0.1:9222')
            
            driver = webdriver.Chrome(options=options)
            print(f"   Connected! Current URL: {driver.current_url}")
            
            # Get cookies from driver
            api.load_cookies_from_selenium(driver)
            cookies_loaded = True
            
            # Save for future use
            api.export_cookies(str(cookies_file))
            
        except Exception as e:
            print(f"   Failed: {e}")
    
    # Method 3: Try browser cookie database
    if not cookies_loaded:
        for browser in ["chrome", "firefox"]:
            print(f"🌐 Trying {browser} cookie database...")
            try:
                api.load_cookies_from_browser(browser)
                cookies_loaded = True
                break
            except Exception as e:
                print(f"   Failed: {e}")
    
    if not cookies_loaded:
        print("\n❌ Could not load cookies from any source!")
        print("\nTo fix this, you can:")
        print("1. Start Chrome with: google-chrome --remote-debugging-port=9222")
        print("2. Log into GeoGuessr in Chrome")
        print("3. Run this script again")
        print("\nOr manually export cookies to cookies.json")
        return 1
    
    # Check authentication
    print("\n🔐 Checking authentication...")
    if api.is_authenticated():
        user_info = api.get_user_info()
        if user_info:
            nick = user_info.get("user", {}).get("nick", "Unknown")
            print(f"   ✅ Authenticated as: {nick}")
    else:
        print("   ⚠️ Not authenticated (cookies may be expired)")
    
    # Extract game ID and type
    game_id = api._get_game_id_from_url(game_url)
    game_type = api._get_game_type_from_url(game_url)
    
    print(f"\n🎮 Game ID: {game_id}")
    print(f"   Type: {game_type}")
    
    # Get game info
    print("\n📊 Getting game info...")
    game_info = api.get_game_info(game_id, game_type)
    if game_info:
        current_round = game_info.get("round", "?")
        total_rounds = game_info.get("roundCount", "?")
        print(f"   Current round: {current_round}/{total_rounds}")
    else:
        print("   Could not get game info (might need auth)")
    
    # Submit guess
    print(f"\n🎯 Submitting guess...")
    result = api.submit_guess(
        game_id=game_id,
        lat=lat,
        lng=lng,
        round_number=round_num,
        game_type=game_type
    )
    
    if result.success:
        print("\n" + "=" * 50)
        print("🎉 GUESS SUBMITTED SUCCESSFULLY!")
        print("=" * 50)
        print(f"   Score: {result.score} points")
        print(f"   Distance: {result.distance_meters:.2f} meters ({result.distance_meters/1000:.2f} km)")
        print(f"   True location: ({result.true_lat:.6f}, {result.true_lng:.6f})")
        if result.total_score:
            print(f"   Total score: {result.total_score}")
        return 0
    else:
        print(f"\n❌ Guess failed: {result.error}")
        if result.raw_response:
            print(f"   Raw response: {json.dumps(result.raw_response, indent=2)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
