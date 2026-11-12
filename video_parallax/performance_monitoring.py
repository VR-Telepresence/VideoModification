import time
from pathlib import Path

from rich.live import Live
from rich.text import Text

if __name__ == "__main__":
    path = Path("./temp/performance.txt")
    print("Working directory:", Path.cwd())
    print("Looking for:", path.resolve())
    print("File exists:", path.exists())

    with Live("", refresh_per_second=5) as live:
        try:
            while True:
                if path.exists():
                    try:
                        output = path.read_text(encoding="utf-8" )
                        live.update(Text(output, no_wrap=True))
                    except PermissionError:
                        """sometimes the main thread is editing the file which leads to permission errors, they can just be ignored as it does not happen to often. 
                        The performance will still be updated regularly.
                        The pass prevents the monitoring program from crashing."""
                        pass
                else:
                    live.update("Waiting for performance data...")

                time.sleep(0.2)

        except KeyboardInterrupt:
            pass