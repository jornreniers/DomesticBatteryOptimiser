import requests
import os
import time
import logging
import numpy as np

from bs4 import BeautifulSoup
from datetime import date
from pathlib import Path
from urllib.parse import urljoin, urlparse

from config.DataConfig import DataConfig

logger = logging.getLogger()

"""
This file is made with AI.
"""


# Configuration
BASE_URL = "https://weatherstationdata.physics.ox.ac.uk"


def _download_file(url, save_dir) -> int:
    """Download a single file with progress indication"""
    try:
        # Get filename from URL
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(save_dir, filename)

        # Skip if file already exists
        if os.path.exists(filepath):
            # logger.debug(f"Skipping (already exists): {filename}")
            return 0

        # Download with streaming
        logger.debug(f"\t\t Downloading: {filename}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Get file size if available
        total_size = int(response.headers.get("content-length", 0))

        # Write to file
        with open(filepath, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Simple progress indicator
                        # percent = (downloaded / total_size) * 100
                        # logger.debug(f"\t Progress: {percent:.1f}%", end="\r")

        # logger.debug(f"Completed: {filename}")
        return 1

    except Exception as e:
        logger.warning(f"\t Error downloading {url}: {str(e)}")
        return -1


def _get_file_links(base_url):
    """Parse directory listing and extract file links"""
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all links
        links = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href not in ["/", "../", ".."]:
                # Skip directories (usually end with /)
                if not href.endswith("/"):
                    full_url = urljoin(base_url, href)  # ty:ignore[invalid-argument-type]
                    links.append(full_url)

        return links

    except Exception as e:
        logger.warning(f"\t Error fetching directory listing: {str(e)}")
        return []


def _download_data(save_dir: str):
    # Create download directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # progress
    successful = 0
    failed = 0
    skipped = 0

    # There is a different webpage for each year
    for year in np.arange(
        DataConfig.weather_start_year, DataConfig.weather_end_year + 1, 1
    ):
        url = BASE_URL + "/" + str(year) + "/"
        logger.debug(f"Downloading data from: {url}\n")

        # Get list of files
        file_links = _get_file_links(url)
        logger.debug(f"\t Found {len(file_links)} files\n")

        # Download each file
        for i, url in enumerate(file_links, 1):
            logger.debug(f"\t Progress for {year}: [{i}/{len(file_links)}]")
            res = _download_file(url, save_dir)
            if res == 1:
                successful += 1
                if i < len(file_links):
                    time.sleep(0.2)  # don't overload internet connection
            elif res == -1:
                failed += 1
            else:
                skipped += 1

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Download Summary:")
    logger.info(f"   Skipped: {skipped}")
    logger.info(f"   Successful: {successful}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Saved to: {os.path.abspath(save_dir)}")
    logger.info("=" * 50)


def _should_run(last_run_file: str):
    """
    Check the last date we downloaded data.
    Only download data if we have not yet run today.
    """
    today = date.today().isoformat()

    # Check if file exists and read last run date
    if os.path.exists(last_run_file):
        with open(last_run_file, "r") as f:
            last_run = f.read().strip()
            if last_run == today:
                return False

    return True


def run(save_dir: str):
    last_run_file = save_dir + "/last_run.txt"
    if _should_run(last_run_file):
        _download_data(save_dir)
        with open(last_run_file, "w") as f:
            f.write(date.today().isoformat())
