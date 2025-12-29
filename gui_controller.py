#!/usr/bin/env python3
"""
Minimal GUI controller for GeoGuessr API Bot
"""

import os
import sys
import csv
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
from typing import Optional

# Import duels bot
try:
    from duels_bot import DuelsBot, DuelState, RoundState
    DUELS_BOT_AVAILABLE = True
except ImportError as e:
    DUELS_BOT_AVAILABLE = False
    print(f"⚠️ DuelsBot not available: {e}")


class TextRedirector:
    """Redirect stdout/stderr to a text widget"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
        
    def write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string, self.tag)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        self.text_widget.update_idletasks()
        
    def flush(self):
        pass


class GeoBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GeoGuessr Bot")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Bot state
        self.duels_bot: Optional[DuelsBot] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.should_stop = threading.Event()
        self.waiting_for_confirmation = threading.Event()
        self.confirmation_received = threading.Event()
        
        # Results tracking
        self.results_file: Optional[str] = None
        self.results_data = []
        
        # Modern color scheme
        self.colors = {
            'bg': '#0f0f0f',
            'surface': '#1a1a1a',
            'surface_hover': '#252525',
            'border': '#333333',
            'text': '#ffffff',
            'text_secondary': '#888888',
            'accent': '#6366f1',
            'accent_hover': '#4f46e5',
            'success': '#22c55e',
            'error': '#ef4444',
            'warning': '#f59e0b',
        }
        
        # Variables
        self.chrome_port_var = tk.StringVar(value="9223")
        self.ml_port_var = tk.StringVar(value="5000")
        self.game_url_var = tk.StringVar(value="")
        self.rounds_var = tk.StringVar(value="5")
        self.results_dir_var = tk.StringVar(value="results")
        
        self.root.configure(bg=self.colors['bg'])
        self.setup_ui()
        
        # Bind Enter key for confirmation
        self.root.bind('<Return>', self.on_enter_pressed)
        
    def setup_ui(self):
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=24, pady=20)
        
        # Header
        header = tk.Frame(main, bg=self.colors['bg'])
        header.pack(fill=tk.X, pady=(0, 20))
        
        title = tk.Label(header, text="GeoGuessr Bot", font=('Helvetica', 24, 'bold'),
                        bg=self.colors['bg'], fg=self.colors['text'])
        title.pack(side=tk.LEFT)
        
        subtitle = tk.Label(header, text="ML-Powered API Bot", font=('Helvetica', 12),
                           bg=self.colors['bg'], fg=self.colors['text_secondary'])
        subtitle.pack(side=tk.LEFT, padx=(12, 0), pady=(8, 0))
        
        # Settings panel
        settings = tk.Frame(main, bg=self.colors['surface'], highlightbackground=self.colors['border'],
                           highlightthickness=1)
        settings.pack(fill=tk.X, pady=(0, 16))
        
        settings_inner = tk.Frame(settings, bg=self.colors['surface'])
        settings_inner.pack(fill=tk.X, padx=16, pady=16)
        
        # Row 1: Ports
        row1 = tk.Frame(settings_inner, bg=self.colors['surface'])
        row1.pack(fill=tk.X, pady=(0, 12))
        
        self._create_input(row1, "Chrome Port", self.chrome_port_var, width=8)
        self._create_input(row1, "ML API Port", self.ml_port_var, width=8)
        self._create_input(row1, "Rounds", self.rounds_var, width=6)
        
        # Row 2: Game URL
        row2 = tk.Frame(settings_inner, bg=self.colors['surface'])
        row2.pack(fill=tk.X, pady=(0, 12))
        
        url_label = tk.Label(row2, text="Game URL", font=('Helvetica', 11),
                            bg=self.colors['surface'], fg=self.colors['text_secondary'])
        url_label.pack(side=tk.LEFT)
        
        url_entry = tk.Entry(row2, textvariable=self.game_url_var, font=('Helvetica', 11),
                            bg=self.colors['bg'], fg=self.colors['text'], insertbackground=self.colors['text'],
                            relief='flat', highlightbackground=self.colors['border'], highlightthickness=1)
        url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0), ipady=6)
        
        # Action buttons
        buttons = tk.Frame(settings_inner, bg=self.colors['surface'])
        buttons.pack(fill=tk.X)
        
        self.start_btn = self._create_button(buttons, "▶  Start Bot", self.start_bot,
                                             bg=self.colors['accent'], hover=self.colors['accent_hover'])
        self.start_btn.pack(side=tk.LEFT, padx=(0, 8))
        
        self.next_btn = self._create_button(buttons, "→  Next Round", self.confirm_next_round,
                                            bg=self.colors['success'], hover='#16a34a')
        self.next_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.next_btn.configure(state='disabled')
        
        self.stop_btn = self._create_button(buttons, "■  Stop", self.stop_bot,
                                            bg=self.colors['error'], hover='#dc2626')
        self.stop_btn.pack(side=tk.LEFT)
        self.stop_btn.configure(state='disabled')
        
        # Status indicator
        self.status_label = tk.Label(buttons, text="● Ready", font=('Helvetica', 11),
                                     bg=self.colors['surface'], fg=self.colors['text_secondary'])
        self.status_label.pack(side=tk.RIGHT)
        
        # Paned window for terminal and results
        paned = tk.PanedWindow(main, orient=tk.VERTICAL, bg=self.colors['border'],
                               sashwidth=4, sashrelief='flat')
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Terminal output
        terminal_frame = tk.Frame(paned, bg=self.colors['surface'], highlightbackground=self.colors['border'],
                                  highlightthickness=1)
        
        terminal_header = tk.Frame(terminal_frame, bg=self.colors['surface'])
        terminal_header.pack(fill=tk.X, padx=12, pady=(12, 8))
        
        tk.Label(terminal_header, text="Terminal", font=('Helvetica', 11, 'bold'),
                bg=self.colors['surface'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.terminal = tk.Text(terminal_frame, font=('JetBrains Mono', 10), wrap=tk.WORD,
                               bg='#0a0a0a', fg='#e4e4e7', relief='flat', padx=12, pady=8,
                               insertbackground=self.colors['text'], state='disabled')
        self.terminal.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        
        # Configure tags for colored output
        self.terminal.tag_config("info", foreground='#e4e4e7')
        self.terminal.tag_config("success", foreground='#4ade80')
        self.terminal.tag_config("error", foreground='#f87171')
        self.terminal.tag_config("warning", foreground='#fbbf24')
        self.terminal.tag_config("highlight", foreground='#a5b4fc')
        
        paned.add(terminal_frame, minsize=150)
        
        # Results table
        results_frame = tk.Frame(paned, bg=self.colors['surface'], highlightbackground=self.colors['border'],
                                 highlightthickness=1)
        
        results_header = tk.Frame(results_frame, bg=self.colors['surface'])
        results_header.pack(fill=tk.X, padx=12, pady=(12, 8))
        
        tk.Label(results_header, text="Results", font=('Helvetica', 11, 'bold'),
                bg=self.colors['surface'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        self.summary_label = tk.Label(results_header, text="", font=('Helvetica', 10),
                                      bg=self.colors['surface'], fg=self.colors['text_secondary'])
        self.summary_label.pack(side=tk.RIGHT)
        
        # Treeview for results
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Treeview",
                       background='#0a0a0a',
                       foreground='#e4e4e7',
                       fieldbackground='#0a0a0a',
                       borderwidth=0,
                       font=('Helvetica', 10))
        style.configure("Custom.Treeview.Heading",
                       background=self.colors['surface'],
                       foreground=self.colors['text'],
                       borderwidth=0,
                       font=('Helvetica', 10, 'bold'))
        style.map("Custom.Treeview", background=[('selected', self.colors['accent'])])
        
        tree_container = tk.Frame(results_frame, bg='#0a0a0a')
        tree_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        
        columns = ('round', 'predicted', 'true_loc', 'distance', 'score')
        self.results_tree = ttk.Treeview(tree_container, columns=columns, show='headings',
                                         style="Custom.Treeview", height=6)
        
        self.results_tree.heading('round', text='Round')
        self.results_tree.heading('predicted', text='Predicted')
        self.results_tree.heading('true_loc', text='True Location')
        self.results_tree.heading('distance', text='Distance')
        self.results_tree.heading('score', text='Score')
        
        self.results_tree.column('round', width=60, anchor='center')
        self.results_tree.column('predicted', width=180, anchor='center')
        self.results_tree.column('true_loc', width=180, anchor='center')
        self.results_tree.column('distance', width=100, anchor='center')
        self.results_tree.column('score', width=80, anchor='center')
        
        scrollbar = ttk.Scrollbar(tree_container, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        paned.add(results_frame, minsize=120)
        
        self.log("Ready. Enter game URL and click Start Bot.", "info")
        
    def _create_input(self, parent, label, var, width=12):
        """Create a labeled input field"""
        frame = tk.Frame(parent, bg=self.colors['surface'])
        frame.pack(side=tk.LEFT, padx=(0, 24))
        
        lbl = tk.Label(frame, text=label, font=('Helvetica', 10),
                      bg=self.colors['surface'], fg=self.colors['text_secondary'])
        lbl.pack(anchor='w')
        
        entry = tk.Entry(frame, textvariable=var, font=('Helvetica', 11), width=width,
                        bg=self.colors['bg'], fg=self.colors['text'], insertbackground=self.colors['text'],
                        relief='flat', highlightbackground=self.colors['border'], highlightthickness=1)
        entry.pack(pady=(4, 0), ipady=4)
        
    def _create_button(self, parent, text, command, bg, hover):
        """Create a styled button"""
        btn = tk.Button(parent, text=text, command=command, font=('Helvetica', 11),
                       bg=bg, fg='white', relief='flat', cursor='hand2',
                       padx=16, pady=8, activebackground=hover, activeforeground='white')
        btn.bind('<Enter>', lambda e: btn.configure(bg=hover))
        btn.bind('<Leave>', lambda e: btn.configure(bg=bg))
        btn._bg = bg
        btn._hover = hover
        return btn
    
    def log(self, message, level="info"):
        """Add message to terminal"""
        self.terminal.configure(state='normal')
        self.terminal.insert(tk.END, message + "\n", level)
        self.terminal.see(tk.END)
        self.terminal.configure(state='disabled')
        
    def set_status(self, text, color=None):
        """Update status indicator"""
        self.status_label.configure(text=f"● {text}", fg=color or self.colors['text_secondary'])
        
    def on_enter_pressed(self, event):
        """Handle Enter key press for round confirmation"""
        if self.waiting_for_confirmation.is_set():
            self.confirm_next_round()
            
    def confirm_next_round(self):
        """Confirm to proceed with next round"""
        if self.waiting_for_confirmation.is_set():
            self.waiting_for_confirmation.clear()
            self.confirmation_received.set()
            self.next_btn.configure(state='disabled')
            self.log("✓ Confirmed - processing round...", "success")
            
    def update_results_table(self, round_state: 'RoundState'):
        """Add a round result to the table"""
        pred = f"({round_state.predicted_lat:.4f}, {round_state.predicted_lng:.4f})" if round_state.predicted_lat else "-"
        true_loc = f"({round_state.lat:.4f}, {round_state.lng:.4f})" if round_state.lat else "-"
        dist = f"{round_state.distance_meters/1000:.2f} km" if round_state.distance_meters else "-"
        score = str(round_state.score) if round_state.score else "-"
        
        self.results_tree.insert('', 'end', values=(
            round_state.round_number, pred, true_loc, dist, score
        ))
        
        # Update summary
        total_score = sum(r.score or 0 for r in self.results_data)
        avg_dist = sum((r.distance_meters or 0) for r in self.results_data) / len(self.results_data) if self.results_data else 0
        self.summary_label.configure(text=f"Total: {total_score} pts | Avg: {avg_dist/1000:.1f} km")
        
    def save_result_to_csv(self, round_state: 'RoundState'):
        """Save round result to CSV file"""
        if not self.results_file:
            results_dir = Path(self.results_dir_var.get())
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_file = results_dir / f"game_results_{timestamp}.csv"
            
            # Write header
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['round', 'predicted_lat', 'predicted_lng', 'true_lat', 'true_lng', 
                               'distance_km', 'score', 'timestamp'])
        
        # Append result
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_state.round_number,
                round_state.predicted_lat,
                round_state.predicted_lng,
                round_state.lat,
                round_state.lng,
                round_state.distance_meters / 1000 if round_state.distance_meters else None,
                round_state.score,
                round_state.timestamp
            ])
            
    def start_bot(self):
        """Start the duels bot"""
        if not DUELS_BOT_AVAILABLE:
            messagebox.showerror("Error", "DuelsBot module not available!")
            return
            
        # Validate inputs
        try:
            chrome_port = int(self.chrome_port_var.get())
            ml_port = int(self.ml_port_var.get())
            rounds = int(self.rounds_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid port or rounds number!")
            return
            
        game_url = self.game_url_var.get().strip()
        ml_api_url = f"http://127.0.0.1:{ml_port}/api/v1/predict"
        
        # Reset state
        self.should_stop.clear()
        self.results_file = None
        self.results_data = []
        
        # Clear results table
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.summary_label.configure(text="")
        
        # Update UI
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self.set_status("Starting...", self.colors['warning'])
        
        def run_bot():
            # Redirect output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = TextRedirector(self.terminal, "info")
            sys.stderr = TextRedirector(self.terminal, "error")
            
            try:
                self.log(f"{'='*50}", "highlight")
                self.log("🤖 Starting GeoGuessr Bot", "highlight")
                self.log(f"{'='*50}", "highlight")
                self.log(f"   Chrome Port: {chrome_port}")
                self.log(f"   ML API: {ml_api_url}")
                self.log(f"   Rounds: {rounds}")
                
                # Create bot
                self.duels_bot = DuelsBot(
                    chrome_debug_port=chrome_port,
                    ml_api_url=ml_api_url,
                    use_screenshot=True,
                )
                
                # Connect to Chrome
                self.set_status("Connecting...", self.colors['warning'])
                if not self.duels_bot.connect_to_chrome():
                    self.log("❌ Failed to connect to Chrome!", "error")
                    self.log(f"\nStart Chrome with:", "warning")
                    self.log(f"  google-chrome --remote-debugging-port={chrome_port} --user-data-dir=/tmp/bot-profile", "info")
                    return
                    
                self.log("✓ Connected to Chrome", "success")
                
                # Navigate to game if URL provided
                if game_url:
                    self.log(f"🌐 Navigating to {game_url}")
                    self.duels_bot.driver.get(game_url)
                    import time
                    time.sleep(3)
                
                # Get game info
                game_id = self.duels_bot.get_game_id_from_url()
                game_type = self.duels_bot.get_game_type_from_url()
                
                if not game_id:
                    self.log("❌ Could not determine game ID from URL", "error")
                    return
                    
                self.log(f"   Game ID: {game_id}")
                self.log(f"   Game Type: {game_type}")
                
                self.set_status("Running", self.colors['success'])
                
                # Set up confirmation callback for the bot
                gui_self = self
                
                def wait_for_gui_confirmation():
                    """Called by bot to wait for user confirmation"""
                    gui_self.log("\n⏳ Press ENTER or click 'Next Round' when ready...", "warning")
                    gui_self.root.after(0, lambda: gui_self.next_btn.configure(state='normal'))
                    gui_self.root.after(0, lambda: gui_self.set_status("Waiting for confirmation", gui_self.colors['warning']))
                    
                    gui_self.waiting_for_confirmation.set()
                    gui_self.confirmation_received.clear()
                    
                    while not gui_self.confirmation_received.is_set():
                        if gui_self.should_stop.is_set():
                            break
                        import time
                        time.sleep(0.1)
                    
                    gui_self.root.after(0, lambda: gui_self.next_btn.configure(state='disabled'))
                    gui_self.root.after(0, lambda: gui_self.set_status("Processing...", gui_self.colors['accent']))
                
                # Assign the callback to the bot
                self.duels_bot.wait_for_confirmation = wait_for_gui_confirmation
                
                # Set up round result callback
                def on_round_complete(round_state):
                    if round_state:
                        gui_self.results_data.append(round_state)
                        gui_self.root.after(0, lambda rs=round_state: gui_self.update_results_table(rs))
                        gui_self.save_result_to_csv(round_state)
                        
                        if round_state.score:
                            gui_self.log(f"✓ Score: {round_state.score} pts", "success")
                        if round_state.distance_meters:
                            gui_self.log(f"   Distance: {round_state.distance_meters/1000:.2f} km", "info")
                
                # Play game using the bot's play_game method
                result = self.duels_bot.play_game(game_url=game_url if game_url else None, num_rounds=rounds)
                
                # Process results from the game
                if result:
                    for round_state in result.rounds:
                        if round_state not in self.results_data:
                            self.results_data.append(round_state)
                            self.root.after(0, lambda rs=round_state: self.update_results_table(rs))
                            self.save_result_to_csv(round_state)
                
                # Game complete
                total = sum(r.score or 0 for r in self.results_data)
                self.log(f"\n{'='*50}", "success")
                self.log(f"🏆 GAME COMPLETE - Total: {total} pts", "success")
                self.log(f"{'='*50}", "success")
                
                if self.results_file:
                    self.log(f"📁 Results saved to: {self.results_file}", "info")
                    
            except Exception as e:
                self.log(f"❌ Error: {e}", "error")
                import traceback
                self.log(traceback.format_exc(), "error")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                if self.duels_bot:
                    self.duels_bot.close()
                    self.duels_bot = None
                    
                self.root.after(0, self._reset_ui)
                
        self.bot_thread = threading.Thread(target=run_bot, daemon=True)
        self.bot_thread.start()
        
    def _reset_ui(self):
        """Reset UI after bot stops"""
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.next_btn.configure(state='disabled')
        self.set_status("Ready", self.colors['text_secondary'])
        
    def stop_bot(self):
        """Stop the bot"""
        self.should_stop.set()
        self.confirmation_received.set()  # Unblock any waiting
        
        if self.duels_bot:
            self.duels_bot.stop()
            
        self.log("\n⚠️ Stopping bot...", "warning")
        self.set_status("Stopping...", self.colors['error'])


def main():
    root = tk.Tk()
    app = GeoBotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

