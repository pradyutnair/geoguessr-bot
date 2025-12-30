#!/usr/bin/env python3
"""Debug API response structure for duels"""
import json
import sys
sys.path.insert(0, '/home/pradyutnair/geoguessr-bot')

from geoguessr_api import GeoGuessrAPI

api = GeoGuessrAPI()
api.load_cookies_from_file('cookies.json')

# Get user info to find our player ID
print("=" * 60)
print("USER INFO")
print("=" * 60)
user_info = api.get_user_info()
if user_info:
    print(json.dumps(user_info, indent=2, default=str)[:2000])
    
    # Look for our ID
    user = user_info.get("user", {})
    our_id = user.get("id")
    our_nick = user.get("nick")
    print(f"\nOur ID: {our_id}")
    print(f"Our Nick: {our_nick}")

# Test with a specific game ID
game_id = input("\nEnter game ID (or press Enter to skip): ").strip()

if game_id:
    print("\n" + "=" * 60)
    print("GAME DATA")
    print("=" * 60)
    
    url = f"https://game-server.geoguessr.com/api/duels/{game_id}"
    response = api.session.get(url, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        
        # Show key parts
        print("\nTop-level keys:", list(data.keys()))
        print(f"\nstate: {data.get('state')}")
        print(f"currentRoundNumber: {data.get('currentRoundNumber')}")
        
        # Check for teams/players
        if 'teams' in data:
            print("\nTeams:")
            for i, team in enumerate(data['teams']):
                print(f"  Team {i}: id={team.get('id')}, health={team.get('health')}")
                for p in team.get('players', []):
                    print(f"    Player: id={p.get('playerId')}, nick={p.get('nick')}")
        
        if 'players' in data:
            print("\nPlayers:")
            for p in data['players']:
                print(f"  {p.get('id') or p.get('playerId')}: nick={p.get('nick')}, health={p.get('health')}")
        
        # Check guesses structure
        guesses = data.get('guesses', {})
        print(f"\nGuesses type: {type(guesses)}")
        
        if isinstance(guesses, dict):
            for player_id, player_guesses in guesses.items():
                print(f"\nPlayer {player_id}:")
                if isinstance(player_guesses, list) and player_guesses:
                    # Show first guess structure
                    first = player_guesses[0]
                    print(f"  Keys: {list(first.keys())}")
                    print(f"  Sample guess: {json.dumps(first, indent=4, default=str)}")
        
        # Save full response for inspection
        with open('debug_game_response.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print("\nFull response saved to debug_game_response.json")
    else:
        print(f"Failed: {response.status_code}")
        print(response.text[:500])
