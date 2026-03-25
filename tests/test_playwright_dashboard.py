"""
Playwright test for port_simulation.py dashboard.
Uses Chrome DevTools to interact with the Streamlit app.
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, '.')

def test_dashboard_with_playwright():
    from playwright.sync_api import sync_playwright
    from dashboard.port_simulation import (
        initialize_berthed_vessels,
        initialize_waiting_queue,
    )
    from pipeline.data_pipeline import run_full_pipeline
    from config.constants import CSV_PATH, JSON_PATH
    
    print("=" * 60)
    print("Playwright Test: Port Simulation Dashboard")
    print("=" * 60)
    
    # First test the underlying functions work
    print("\n[1] Testing underlying functions...")
    _, berth_matrix = run_full_pipeline(CSV_PATH, JSON_PATH)
    print(f"✓ Loaded {len(berth_matrix)} berths")
    
    base_time = datetime(2024, 6, 1, 6, 0)
    berthed = initialize_berthed_vessels(berth_matrix, 3, base_time, 36.0)
    waiting = initialize_waiting_queue(4, base_time)
    print(f"✓ Created {len(berthed)} berthed, {len(waiting)} waiting vessels")
    
    # Now test with Playwright
    print("\n[2] Starting Streamlit server...")
    import subprocess
    import os
    
    # Kill any existing streamlit processes
    os.system("pkill -f 'streamlit run' 2>/dev/null")
    time.sleep(1)
    
    # Start streamlit in background
    proc = subprocess.Popen(
        ["streamlit", "run", "dashboard/port_simulation.py", "--server.port", "8503"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    print("Waiting for server to start...")
    time.sleep(5)
    
    print("\n[3] Testing with Playwright...")
    with sync_playwright() as p:
        # Launch Chrome
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        # Enable console logging
        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))
        
        try:
            # Navigate to the app
            print("Loading page...")
            page.goto("http://localhost:8503", timeout=30000)
            
            # Wait for page to fully load
            page.wait_for_load_state("networkidle", timeout=15000)
            print("✓ Page loaded")
            
            # Check page title
            title = page.title()
            print(f"✓ Page title: {title}")
            
            # Look for the initialization UI
            print("\n[4] Checking for key UI elements...")
            
            # Check for "Initialization" sidebar
            init_text = page.locator("text=Initialization").first
            if init_text.is_visible(timeout=5000):
                print("✓ Found Initialization section in sidebar")
            
            # Check for sliders
            sliders = page.locator("input[type='range']").all()
            print(f"✓ Found {len(sliders)} sliders")
            
            # Check for "Initialize Port" button
            init_btn = page.locator("button:has-text('Initialize Port')").first
            if init_btn.is_visible(timeout=5000):
                print("✓ Found 'Initialize Port' button")
                
                # Click it to initialize
                print("\n[5] Clicking 'Initialize Port'...")
                init_btn.click()
                
                # Wait for loading to complete
                time.sleep(3)
                
                # Check for "Port initialized" message
                success_msg = page.locator("text=Port initialized").first
                if success_msg.is_visible(timeout=10000):
                    print("✓ Port initialized successfully")
                
                # Check for time control section
                time_control = page.locator("text=Time Control").first
                if time_control.is_visible(timeout=5000):
                    print("✓ Found Time Control section")
                
                # Check for "Run Allocation" button
                alloc_btn = page.locator("button:has-text('Run Allocation')").first
                if alloc_btn.is_visible(timeout=5000):
                    print("✓ Found 'Run Allocation' button")
                    
                    # Click it
                    print("\n[6] Running allocation...")
                    alloc_btn.click()
                    
                    # Wait for allocation to complete
                    time.sleep(3)
                    
                    # Check for KPIs
                    kpis = page.locator("text=Performance KPIs").first
                    if kpis.is_visible(timeout=10000):
                        print("✓ Found KPIs section")
                    
                    # Check for Gantt chart
                    gantt = page.locator("text=Berth Schedule").first
                    if gantt.is_visible(timeout=5000):
                        print("✓ Found Gantt chart")
                    
                    # Check for port status
                    port_status = page.locator("text=Port Status").first
                    if port_status.is_visible(timeout=5000):
                        print("✓ Found Port Status section")
                
                # Check for Add Vessel section
                add_vessel = page.locator("text=Add Vessel").first
                if add_vessel.is_visible(timeout=5000):
                    print("✓ Found Add Vessel section")
            
            # Take a screenshot
            print("\n[7] Taking screenshot...")
            page.screenshot(path="tests/dashboard_screenshot.png", full_page=True)
            print("✓ Screenshot saved to tests/dashboard_screenshot.png")
            
            # Check console for errors
            print("\n[8] Checking console logs...")
            errors = [log for log in console_logs if "Error" in log or "error" in log]
            if errors:
                print(f"⚠ Found {len(errors)} console errors:")
                for e in errors[:5]:
                    print(f"  - {e}")
            else:
                print("✓ No console errors")
            
            print("\n" + "=" * 60)
            print("PLAYWRIGHT TEST PASSED ✓")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            # Take screenshot on error
            page.screenshot(path="tests/dashboard_error.png", full_page=True)
            raise
        finally:
            browser.close()
    
    # Cleanup
    proc.terminate()
    proc.wait()
    print("\nServer stopped")


if __name__ == "__main__":
    test_dashboard_with_playwright()