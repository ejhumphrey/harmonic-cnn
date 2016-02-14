"""Walk a website collecting links, and saving off a subset of "resources".

Example:
$ python2 spider.py \
    http://theremin.music.uiowa.edu/ \
    uiowa_mis.json \
    --domain_filter theremin.music.uiowa.edu \
    --delay 0.5
"""

import argparse
from bs4 import BeautifulSoup
import logging
import httplib
from htmllib import HTMLParseError
import json
import numpy as np
import os
import requests
import time
import urlparse
import yurl


RESOURCES = ['.jpg', '.zip', '.sty', '.ppt', '.doc', '.aiff', '.mp3', '.aif',
             '.tex', '.png', '.txt', '.pdf', '.gz', '']

GLOBS = dict(last_time=time.time())
logger = logging.getLogger(__name__)


def find_urls(soup):
    """Pluck URLs from the soup.

    Note: these URLs may be relative.

    Parameters
    ----------
    soup : bs4.Soup
        HTML page soup.

    Returns
    -------
    urls : list of str
        URLs on the page.
    """
    next_urls = list()
    for link in soup.findAll("a"):
        url = link.attrs.get("href", '').strip()
        if url not in next_urls and url:
            next_urls.append(url)

    return next_urls


def is_valid_url(url):
    """Test that a URL is alive.

    Parameters
    ----------
    url : str
        URL to test.

    Returns
    -------
    stat : bool
        True if valid.
    """
    p = urlparse.urlparse(url)
    conn = httplib.HTTPConnection(p.netloc)
    conn.request('HEAD', p.path)
    resp = conn.getresponse()
    return resp.status < 400


def is_http(url):
    """Test that a URL is an HTTP scheme.

    Parameters
    ----------
    url : str
        URL to test.

    Returns
    -------
    stat : bool
        True if valid.
    """
    return urlparse.urlparse(url).scheme == 'http'


def is_resource(url):
    """Test that a URL is one of the resources of interest.

    Parameters
    ----------
    url : str
        URL to test.

    Returns
    -------
    stat : bool
        True if valid.
    """
    path = urlparse.urlparse(url).path
    return os.path.splitext(path)[-1].lower() in RESOURCES


def expand_url(next_url, base_url):
    """Expand a given URL to its base, if relative.

    Parameters
    ----------
    next_url : str
        URL to escape.

    base_url : str
        Base URL of the URL of interest.

    Returns
    -------
    full_url : str
        Expanded URL.
    """
    next_yurl = yurl.URL(next_url)
    if next_yurl.is_relative():
        next_url = urlparse.urljoin(base_url, next_url)
    return next_url


def request_soup(url):
    """Get the soup-version of a URL.

    Parameters
    ----------
    url : str
        Page to soupify.

    Returns
    -------
    soup : bs4.Soup or None
        Soup version of the page, or None on death.
    """
    try:
        req = requests.get(url)
    except requests.exceptions.ConnectionError:
        logger.error("Failed HTTP Request (Timeout): {}".format(url))
        return None
    if req.status_code >= 400:
        logger.error("Failed HTTP Request (Doesn't exist): {}".format(url))
        return None
    try:
        soup = BeautifulSoup(req.text)
    except HTMLParseError:
        logger.error("BeautifulSoup Error (HTML Parse): {}".format(url))
        return None
    return soup


def get_next_urls(base_url, domain_filter=''):
    """Get URLs referenced from a base URL.

    Parameters
    ----------
    base_url : str
        Base URL to traverse.

    domain_filter : str, default=''
        If given, will limit URLs to this domain.

    Returns
    -------
    next_urls : list of str
        Collection of URLs found on the page.
    """
    soup = request_soup(base_url)
    if not soup:
        return list()
    found_urls = find_urls(soup)
    next_urls = []
    for next_url in found_urls:
        full_url = expand_url(next_url, base_url)
        if domain_filter in urlparse.urlparse(full_url).netloc:
            next_urls.append((base_url, full_url))
    return next_urls


def walk(base_url, domain_filter='', delay=0.5, visited=None, resources=None):
    """Walk a given URL, collecting pages and resources.

    Parameters
    ----------
    base_url : str
        Base URL to traverse.

    domain_filter : str, default=''
        If given, will limit URLs to this domain.

    delay : scalar, >0 default=0.5
        Minimum time in seconds to wait between requests, as a courtesy.

    visited, resources : sets, default=None
        Sets of URLs, passed by reference; if None, will be created.

    Returns
    -------
    visited, resources : sets
        Visited URLs, and the subset of resources, respectively.
    """
    open_urls = [('start', base_url)]
    throttle = True
    visited = set() if visited is None else visited
    resources = set() if resources is None else resources
    while open_urls:
        tdiff = np.abs(time.time() - GLOBS['last_time'])
        if tdiff < delay and throttle:
            time.sleep(min([np.abs(delay - tdiff), delay]))
            throttle = False
        GLOBS['last_time'] = time.time()
        source, this_url = open_urls.pop(0)

        if not is_http(this_url) or this_url in visited:
            continue
        try:
            logger.info("Visited: {}".format(this_url))
            visited.add(this_url)
        except UnicodeEncodeError as derp:
            print(this_url, derp)
            continue

        # Pass over resources
        if is_resource(this_url):
            resources.add(this_url)
        else:
            # Get next urls, if any
            open_urls.extend(get_next_urls(this_url, domain_filter))
            throttle = True
    return visited, resources


def main(base_url, output_file, domain_filter='', delay=0.5):
    if os.path.exists(output_file):
        data = json.load(open(args.output_file))
        visited = set(data['visited'])
        resources = set(data['resources'])
    else:
        visited, resources = set(), set()

    try:
        walk(base_url, domain_filter, delay, visited, resources)
    except KeyboardInterrupt:
        print('Stopping early...')

    results = dict(visited=list(visited), resources=list(resources),
                   base_url=base_url)
    with open(args.output_file, 'w') as fp:
        json.dump(results, fp, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url",
        metavar="url", type=str,
        help="Website to crawl.")
    parser.add_argument(
        "output_file",
        metavar="output_file", type=str,
        help="Filepath for writing collected outputs.")
    parser.add_argument(
        "--domain_filter",
        metavar="domain_filter", type=str, default='',
        help="Domain restriction.")
    parser.add_argument(
        "--delay",
        metavar="delay", type=float, default=0.5,
        help="Time to delay between HTTP requests.")

    args = parser.parse_args()
    main(args.url, args.output_file,
         domain_filter=args.domain_filter, delay=args.delay)
