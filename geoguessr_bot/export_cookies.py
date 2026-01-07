#!/usr/bin/env python3
"""
Export cookies from Chrome browser session to cookies.json

Usage:
1. Make sure Chrome is running with remote debugging:
   google-chrome --remote-debugging-port=9223 --user-data-dir=/tmp/bot-profile

2. Log into GeoGuessr with your new account in that browser

3. Run this script:
   python export_cookies.py
"""

import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def export_cookies(chrome_port: int = 9223, output_file: str = "cookies.json"):
    """Export cookies from Chrome to JSON file."""
    
    print(f"🔌 Connecting to Chrome on port {chrome_port}...")
    
    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{chrome_port}")
    
    try:
        driver = webdriver.Chrome(options=options)
        print(f"✅ Connected to Chrome")
        print(f"   Current URL: {driver.current_url}")
        
        # Navigate to GeoGuessr to ensure we get all cookies
        if "geoguessr.com" not in driver.current_url:
            print("🌐 Navigating to GeoGuessr...")
            driver.get("https://www.geoguessr.com")
            import time
            time.sleep(2)
        
        # Get all cookies
        selenium_cookies = driver.get_cookies()
        
        # Convert to simple dict format
        cookies = {}
        for cookie in selenium_cookies:
            cookies[cookie['name']] = cookie['value']
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        
        print(f"\n✅ Exported {len(cookies)} cookies to {output_file}")
        print(f"\nImportant cookies found:")
        
        # Check for important auth cookies
        important_cookies = ['_ncfa', 'devicetoken', '_cfuvid']
        for name in important_cookies:
            if name in cookies:
                value = cookies[name][:30] + "..." if len(cookies[name]) > 30 else cookies[name]
                print(f"   ✓ {name}: {value}")
            else:
                print(f"   ✗ {name}: NOT FOUND")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print(f"  1. Chrome is running with: --remote-debugging-port={chrome_port}")
        print("  2. You are logged into GeoGuessr in that browser")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export GeoGuessr cookies from Chrome")
    parser.add_argument("--port", "-p", type=int, default=9223, help="Chrome debug port")
    parser.add_argument("--output", "-o", default="cookies.json", help="Output file")
    args = parser.parse_args()
    
    export_cookies(args.port, args.output)
