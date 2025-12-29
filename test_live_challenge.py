#!/usr/bin/env python3
"""
Test script for Live Challenge integration.
Tests URL parsing, browser connection, and API submission.
"""

import sys
from geoguessr_api import GeoGuessrAPI
from duels_bot import DuelsBot

# Test URL
TEST_URL = "https://www.geoguessr.com/live-challenge/ad5900a8-4567-40d5-a817-ef73a4af31c7"

def test_url_parsing():
    """Test that we can extract game ID from live-challenge URL."""
    print("=" * 60)
    print("TEST 1: URL Parsing")
    print("=" * 60)
    
    bot = DuelsBot()
    
    # Test game ID extraction
    game_id = bot.get_game_id_from_url(TEST_URL)
    print(f"URL: {TEST_URL}")
    print(f"Extracted Game ID: {game_id}")
    
    # Test game type detection
    game_type = bot.get_game_type_from_url(TEST_URL)
    print(f"Detected Game Type: {game_type}")
    
    expected_id = "ad5900a8-4567-40d5-a817-ef73a4af31c7"
    expected_type = "live-challenge"
    
    if game_id == expected_id:
        print("✅ Game ID extraction: PASS")
    else:
        print(f"❌ Game ID extraction: FAIL (expected {expected_id})")
        return False
    
    if game_type == expected_type:
        print("✅ Game type detection: PASS")
    else:
        print(f"❌ Game type detection: FAIL (expected {expected_type})")
        return False
    
    return True


def test_api_endpoints():
    """Test that API endpoints are correctly configured."""
    print("\n" + "=" * 60)
    print("TEST 2: API Endpoints")
    print("=" * 60)
    
    api = GeoGuessrAPI()
    
    # Check that live-challenge endpoint exists
    if "live-challenge" in api.API_ENDPOINTS:
        endpoint = api.API_ENDPOINTS["live-challenge"]
        print(f"Live-challenge endpoint: {endpoint}")
        print("✅ Endpoint configured: PASS")
    else:
        print("❌ No live-challenge endpoint defined")
        return False
    
    return True


def test_browser_connection():
    """Test connecting to Chrome with remote debugging."""
    print("\n" + "=" * 60)
    print("TEST 3: Browser Connection (port 9223)")
    print("=" * 60)
    
    bot = DuelsBot(chrome_debug_port=9223)
    
    if bot.connect_to_chrome():
        print(f"✅ Connected to Chrome")
        print(f"   Current URL: {bot.driver.current_url}")
        
        # Try to detect game from current page
        game_id = bot.get_game_id_from_url()
        game_type = bot.get_game_type_from_url()
        
        print(f"   Detected Game ID: {game_id}")
        print(f"   Detected Game Type: {game_type}")
        
        return True, bot
    else:
        print("❌ Failed to connect to Chrome")
        print("   Make sure Chrome is running with:")
        print("   google-chrome --remote-debugging-port=9223 --user-data-dir=/tmp/bot-profile")
        return False, None


def test_api_authentication(bot: DuelsBot):
    """Test that API is authenticated."""
    print("\n" + "=" * 60)
    print("TEST 4: API Authentication")
    print("=" * 60)
    
    if bot.api.is_authenticated():
        user = bot.api.get_user_info()
        if user:
            nick = user.get("nick", "Unknown")
            print(f"✅ Authenticated as: {nick}")
            return True
        else:
            print("⚠️ Authenticated but couldn't get user info")
            return True
    else:
        print("❌ Not authenticated")
        print("   Loading cookies from cookies.json...")
        if bot.api.load_cookies_from_file("cookies.json"):
            print("   Cookies loaded, checking again...")
            if bot.api.is_authenticated():
                print("✅ Now authenticated")
                return True
        return False


def test_get_game_state(bot: DuelsBot, game_id: str):
    """Test fetching game state from API."""
    print("\n" + "=" * 60)
    print("TEST 5: Get Game State")
    print("=" * 60)
    
    print(f"Fetching state for game: {game_id}")
    state = bot.api.get_duel_state(game_id)
    
    if state:
        print("✅ Got game state:")
        print(f"   Current Round: {state.get('currentRoundNumber', 'N/A')}")
        print(f"   Game Status: {state.get('state', 'N/A')}")
        
        # Try to get pano ID
        rounds = state.get("rounds", [])
        if rounds:
            current_round = state.get("currentRoundNumber", 1)
            if current_round <= len(rounds):
                round_data = rounds[current_round - 1]
                pano_id = round_data.get("panoId") or round_data.get("panoramaId")
                print(f"   Pano ID for round {current_round}: {pano_id[:30] if pano_id else 'N/A'}...")
        
        return True, state
    else:
        print("❌ Failed to get game state")
        print("   The game might have ended or you're not a participant")
        return False, None


def test_pano_extraction(bot: DuelsBot):
    """Test extracting panorama ID from browser."""
    print("\n" + "=" * 60)
    print("TEST 6: Panorama Extraction from Browser")
    print("=" * 60)
    
    pano_id = bot.extract_pano_id_from_browser()
    
    if pano_id:
        print(f"✅ Extracted Pano ID: {pano_id[:40]}...")
        return True, pano_id
    else:
        print("⚠️ Could not extract pano ID from browser")
        print("   This might be normal if the round hasn't started yet")
        return False, None


def main():
    print("\n🧪 LIVE CHALLENGE BOT TEST SUITE")
    print("=" * 60)
    print(f"Test URL: {TEST_URL}")
    print("=" * 60)
    
    # Run tests
    all_passed = True
    
    # Test 1: URL Parsing
    if not test_url_parsing():
        all_passed = False
    
    # Test 2: API Endpoints
    if not test_api_endpoints():
        all_passed = False
    
    # Test 3: Browser Connection
    success, bot = test_browser_connection()
    if not success:
        print("\n⚠️ Cannot continue without browser connection")
        print("   Remaining tests skipped")
        return 1
    
    # Test 4: API Authentication
    if not test_api_authentication(bot):
        print("\n⚠️ API not authenticated - some tests may fail")
    
    # Get game ID from test URL
    game_id = bot.get_game_id_from_url(TEST_URL)
    
    # Test 5: Game State
    success, state = test_get_game_state(bot, game_id)
    if not success:
        all_passed = False
    
    # Test 6: Pano Extraction
    success, pano_id = test_pano_extraction(bot)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if all_passed:
        print("✅ All critical tests passed!")
        print("\nYou can now run the duels bot with:")
        print(f"   python duels_bot.py --game-url \"{TEST_URL}\" --chrome-port 9223")
    else:
        print("❌ Some tests failed - check output above")
    
    # Cleanup
    if bot:
        bot.close()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
