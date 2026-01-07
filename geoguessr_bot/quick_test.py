#!/usr/bin/env python3
"""Quick test of guess submission"""
import sys
sys.path.insert(0, '/home/pradyutnair/geoguessr-bot')

from duels_bot import DuelsBot

print("Starting test...")

bot = DuelsBot(chrome_debug_port=9223)
print("Created bot")

if bot.connect_to_chrome():
    print("Connected!")
    game_id = bot.get_game_id_from_url()
    game_type = bot.get_game_type_from_url()
    print(f"Game ID: {game_id}")
    print(f"Game Type: {game_type}")
    
    # Test guess
    print("Submitting guess...")
    result = bot.api.submit_guess(
        game_id=game_id,
        lat=48.8566,
        lng=2.3522,
        round_number=1,
        game_type=game_type
    )
    print(f"Success: {result.success}")
    print(f"Score: {result.score}")
    print(f"Error: {result.error}")
    print(f"Raw: {result.raw_response}")
    bot.close()
else:
    print("Failed to connect")
