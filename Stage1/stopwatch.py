import tkinter as tk
import time



class Stopwatch:
    def __init__(self, master):
        self.master = master

        self.is_running = False
        self.start_time = 0

        self.label = tk.Label(self.master, text="00:00:00", font=("Helvetica", 24))
        self.label.pack(padx=20, pady=20)

        self.start_button = tk.Button(self.master, text="Start", command=self.start_stop)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(self.master, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.RIGHT, padx=10)

        self.update()

    def start_stop(self):
        if self.is_running:
            self.is_running = False
            self.start_button.config(text="Start")
        else:
            self.is_running = True
            self.start_button.config(text="Stop")
            if not self.start_time:  # Start the timer from 0 if it hasn't started yet
                self.start_time = time.time()

    def reset(self):
        self.is_running = False
        self.start_time = 0
        self.label.config(text="00:00:00")
        self.start_button.config(text="Start")

    @property
    def elapsed_time(self):
        if self.is_running:
            return time.time() - self.start_time
        return 0

    def update(self):
        if self.is_running:
            elapsed_seconds = int(self.elapsed_time)
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            seconds = elapsed_seconds % 60
            self.label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        self.master.after(1000, self.update)