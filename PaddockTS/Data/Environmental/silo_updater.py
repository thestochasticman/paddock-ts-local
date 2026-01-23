#!/usr/bin/env python3
"""
Background SILO cache updater service.

This script runs in the background and periodically updates the SILO cache
for the current year. It can be run as:
- A standalone daemon: python silo_updater.py --daemon
- A one-shot update: python silo_updater.py --once
- Via systemd service
- Via cron job

Usage:
    python -m PaddockTS.Data.Environmental.silo_updater --daemon
    python -m PaddockTS.Data.Environmental.silo_updater --once
    python -m PaddockTS.Data.Environmental.silo_updater --once --years 2024 2025
"""
import argparse
import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# Import config for default paths
from PaddockTS.config import config

# Default configuration
DEFAULT_SILO_FOLDER = None  # Will use config.silo_dir
DEFAULT_UPDATE_INTERVAL_HOURS = 6
DEFAULT_VARIABLES = [
    "radiation", "vp", "max_temp", "min_temp", "daily_rain",
    "et_morton_actual", "et_morton_potential"
]
# How many past years to ensure are cached
DEFAULT_PAST_YEARS = 10  # Cache data from 2016 onwards by default

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class SiloUpdater:
    """Background SILO cache updater."""

    def __init__(self, silo_folder: str, variables: list[str] = None,
                 update_interval_hours: float = DEFAULT_UPDATE_INTERVAL_HOURS,
                 past_years: int = DEFAULT_PAST_YEARS):
        self.silo_folder = silo_folder
        self.variables = variables or DEFAULT_VARIABLES
        self.update_interval = update_interval_hours * 3600  # Convert to seconds
        self.past_years = past_years
        self.running = False

    def get_years_to_check(self) -> list[str]:
        """Get list of years to check/update (current year + past years)."""
        current_year = datetime.now().year
        start_year = current_year - self.past_years
        return [str(y) for y in range(start_year, current_year + 1)]

    def find_missing_cache(self) -> list[tuple[str, str]]:
        """Find missing year/variable combinations in cache."""
        missing = []
        os.makedirs(self.silo_folder, exist_ok=True)

        for year in self.get_years_to_check():
            for var in self.variables:
                zarr_path = os.path.join(self.silo_folder, f"{year}.{var}.zarr")
                if not os.path.exists(zarr_path):
                    missing.append((year, var))

        return missing

    def update_once(self, years: list[str] = None, fill_missing: bool = True):
        """Run a single update cycle.

        Parameters
        ----------
        years : list[str], optional
            Specific years to update. If None, updates current year.
        fill_missing : bool
            If True, also download any missing past years data.
        """
        from PaddockTS.Data.Environmental.silo_daily import update_silo_cache

        # First, check for and fill missing past years if requested
        if fill_missing:
            missing = self.find_missing_cache()
            if missing:
                logger.info(f"Found {len(missing)} missing cache entries")
                missing_years = sorted(set(year for year, var in missing))
                logger.info(f"Missing years: {missing_years}")

                # Download missing data
                for year in missing_years:
                    missing_vars = [var for y, var in missing if y == year]
                    logger.info(f"Downloading missing data for {year}: {missing_vars}")
                    try:
                        update_silo_cache(
                            silo_folder=self.silo_folder,
                            variables=missing_vars,
                            years=[year],
                            verbose=True
                        )
                    except Exception as e:
                        logger.error(f"Failed to download {year}: {e}")

        # Then update current year (or specified years)
        if years is None:
            years = [str(datetime.now().year)]

        logger.info(f"Updating SILO cache for years: {years}")
        logger.info(f"Variables: {self.variables}")
        logger.info(f"Cache folder: {self.silo_folder}")

        try:
            update_silo_cache(
                silo_folder=self.silo_folder,
                variables=self.variables,
                years=years,
                verbose=True
            )
            logger.info("SILO cache update completed successfully")
        except Exception as e:
            logger.error(f"SILO cache update failed: {e}")
            raise

    def run_daemon(self):
        """Run as a background daemon with periodic updates."""
        self.running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"SILO updater daemon started")
        logger.info(f"Update interval: {self.update_interval / 3600:.1f} hours")
        logger.info(f"Cache folder: {self.silo_folder}")

        while self.running:
            try:
                self.update_once()
            except Exception as e:
                logger.error(f"Update cycle failed: {e}")

            # Sleep in small increments to allow graceful shutdown
            sleep_remaining = self.update_interval
            while sleep_remaining > 0 and self.running:
                sleep_time = min(60, sleep_remaining)  # Check every minute
                time.sleep(sleep_time)
                sleep_remaining -= sleep_time

        logger.info("SILO updater daemon stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False


def write_systemd_service(silo_folder: str = None, output_path: str = None):
    """Generate a systemd service file for the updater."""
    if output_path is None:
        output_path = os.path.expanduser("~/.config/systemd/user/silo-updater.service")

    if silo_folder is None:
        silo_folder = config.silo_dir

    python_path = sys.executable
    script_path = os.path.abspath(__file__)

    service_content = f"""[Unit]
Description=SILO Cache Updater Service
After=network.target

[Service]
Type=simple
ExecStart={python_path} {script_path} --daemon --silo-folder {silo_folder}
Restart=on-failure
RestartSec=60

[Install]
WantedBy=default.target
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(service_content)

    print(f"Systemd service file written to: {output_path}")
    print("\nTo enable and start the service:")
    print("  systemctl --user daemon-reload")
    print("  systemctl --user enable silo-updater")
    print("  systemctl --user start silo-updater")
    print("\nTo check status:")
    print("  systemctl --user status silo-updater")


def write_cron_entry(silo_folder: str = None):
    """Print cron entry for periodic updates."""
    if silo_folder is None:
        silo_folder = config.silo_dir

    python_path = sys.executable
    script_path = os.path.abspath(__file__)

    print("Add this line to your crontab (crontab -e):")
    print(f"\n# Update SILO cache every 6 hours")
    print(f"0 */6 * * * {python_path} {script_path} --once --silo-folder {silo_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="SILO cache updater service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as a background daemon with periodic updates"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single update and exit"
    )
    parser.add_argument(
        "--silo-folder", type=str, default=None,
        help=f"SILO cache folder (default: from config, currently {config.silo_dir})"
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_UPDATE_INTERVAL_HOURS,
        help=f"Update interval in hours for daemon mode (default: {DEFAULT_UPDATE_INTERVAL_HOURS})"
    )
    parser.add_argument(
        "--years", nargs="+", type=str,
        help="Years to update (default: current year only)"
    )
    parser.add_argument(
        "--variables", nargs="+", type=str,
        help="Variables to update (default: common variables)"
    )
    parser.add_argument(
        "--past-years", type=int, default=DEFAULT_PAST_YEARS,
        help=f"Number of past years to ensure are cached (default: {DEFAULT_PAST_YEARS})"
    )
    parser.add_argument(
        "--no-fill-missing", action="store_true",
        help="Skip filling missing past years data"
    )
    parser.add_argument(
        "--install-systemd", action="store_true",
        help="Generate systemd service file"
    )
    parser.add_argument(
        "--install-cron", action="store_true",
        help="Print cron entry for periodic updates"
    )

    args = parser.parse_args()

    if args.install_systemd:
        write_systemd_service(args.silo_folder or config.silo_dir)
        return

    if args.install_cron:
        write_cron_entry(args.silo_folder or config.silo_dir)
        return

    if not args.daemon and not args.once:
        parser.print_help()
        print("\nError: Please specify --daemon or --once")
        sys.exit(1)

    silo_folder = args.silo_folder or config.silo_dir
    updater = SiloUpdater(
        silo_folder=silo_folder,
        variables=args.variables,
        update_interval_hours=args.interval,
        past_years=args.past_years
    )

    if args.once:
        updater.update_once(years=args.years, fill_missing=not args.no_fill_missing)
    else:
        updater.run_daemon()


if __name__ == "__main__":
    main()
