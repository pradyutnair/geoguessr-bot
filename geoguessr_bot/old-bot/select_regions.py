"""
Interactive tool to select screen regions for the GeoGuessr bot.

Run this script and follow the prompts to position your mouse and press 'a' to record locations.
The coordinates will be saved to screen_regions.yaml.
"""

import yaml
import pyautogui
from pynput import keyboard


PRESS_KEY = "a"


def get_keyboard_position(prompt: str) -> list:
    """Wait for user to press 'a' key and return current mouse position as [x, y] list."""
    print(f"\n👆 {prompt}")
    print(f"   Move mouse to the location and press '{PRESS_KEY}'...")
    
    position = [None]
    
    def on_press(key):
        try:
            if key.char == PRESS_KEY:
                x, y = pyautogui.position()
                position[0] = [x, y]
                print(f"   ✅ Recorded: {position[0]}")
                return False  # Stop listener
        except AttributeError:
            pass
    
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join(timeout=60)
    
    if position[0] is None:
        raise TimeoutError(f"Timeout waiting for '{PRESS_KEY}' key press")
    
    return position[0]


def get_coords(players: int = 1) -> dict:
    """
    Interactive coordinate collection for GeoGuessr bot.
    
    Args:
        players: Number of players (1 for solo, 2 for duels)
    
    Returns:
        Dictionary with all screen regions
    """
    print("\n" + "="*60)
    print("🎮 GeoGuessr Bot - Screen Region Configuration")
    print("="*60)
    print("\nThis tool will help you configure the screen regions.")
    print("Please have GeoGuessr open in your browser with a game started.")
    print("\nWhen prompted, click on the specified location.")
    input("\nPress Enter when ready...")
    
    regions = {}
    
    # Screen capture region
    print("\n📺 SCREEN CAPTURE REGION")
    print("-" * 40)
    regions["screen_top_left"] = get_keyboard_position("Move mouse to TOP-LEFT corner of the game view (panorama area)")
    regions["screen_bot_right"] = get_keyboard_position("Move mouse to BOTTOM-RIGHT corner of the game view")
    
    for player in range(1, players + 1):
        player_str = f" (Player {player})" if players > 1 else ""
        
        # Minimap region - IMPORTANT: calibrate the EXPANDED map
        print(f"\n🗺️ MINIMAP REGION{player_str}")
        print("-" * 40)
        print("   ⚠️  IMPORTANT: First HOVER over the minimap to EXPAND it!")
        print("   Then position mouse on the corners of the EXPANDED map.")
        input("   Press Enter when map is expanded...")
        regions[f"map_top_left_{player}"] = get_keyboard_position(f"Move mouse to TOP-LEFT corner of the EXPANDED minimap{player_str}")
        regions[f"map_bot_right_{player}"] = get_keyboard_position(f"Move mouse to BOTTOM-RIGHT corner of the EXPANDED minimap{player_str}")
        
        # Confirm button
        print(f"\n✅ CONFIRM BUTTON{player_str}")
        print("-" * 40)
        regions[f"confirm_button_{player}"] = get_keyboard_position(f"Move mouse to the CONFIRM/GUESS button{player_str}")
    
    # Calibration points for accurate coordinate mapping
    print("\n🎯 CALIBRATION POINTS")
    print("-" * 40)
    print("   These reference points help calibrate lat/lng to pixel mapping.")
    print("   Make sure the minimap is EXPANDED and zoomed to WORLD VIEW (fully zoomed out).")
    input("   Press Enter when ready for calibration...")
    
    # Kodiak, Alaska calibration point
    print("\n📍 Kodiak, Alaska (57.79°N, -152.41°W)")
    print("   Navigate to Kodiak on the minimap (or search for it)")
    print("   Position mouse on Kodiak and press 'a'")
    kodiak_pixel = get_keyboard_position("Kodiak, Alaska")
    regions["calibration_kodiak"] = {
        "pixel": kodiak_pixel,
        "lat": 57.79,
        "lng": -152.41
    }
    
    # Hobart, Tasmania calibration point
    print("\n📍 Hobart, Tasmania (-42.88°S, 147.33°E)")
    print("   Navigate to Hobart on the minimap (or search for it)")
    print("   Position mouse on Hobart and press 'a'")
    hobart_pixel = get_keyboard_position("Hobart, Tasmania")
    regions["calibration_hobart"] = {
        "pixel": hobart_pixel,
        "lat": -42.88,
        "lng": 147.33
    }
    
    # Next round button (for player 1 only in single player)
    if players == 1:
        print("\n⏭️ NEXT ROUND BUTTON")
        print("-" * 40)
        print("   After guessing, there's sometimes a 'Next Round' or 'Play Next' button.")
        print("   If your game auto-advances with SPACE key, you can skip this.")
        skip = input("   Skip next round button? (y/n): ").lower().strip()
        if skip != 'y':
            regions["next_round_button"] = get_keyboard_position("Move mouse to the NEXT ROUND button (after making a guess)")
        else:
            regions["next_round_button"] = None
    
    # Save to file
    print("\n💾 Saving configuration...")
    with open("screen_regions.yaml", "w") as f:
        yaml.dump(regions, f, default_flow_style=False)
    
    print("✅ Configuration saved to screen_regions.yaml")
    print("\nYou can now run the bot with: python main_bot.py")
    
    return regions


if __name__ == "__main__":
    import sys
    players = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    get_coords(players=players)

