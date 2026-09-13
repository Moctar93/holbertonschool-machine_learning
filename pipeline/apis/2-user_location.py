#!/usr/bin/env python3
"""Get the location of a GitHub user."""

import sys
import time

import requests


if __name__ == '__main__':
    url = sys.argv[1]
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print(response.json()['location'])
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        reset = int(response.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = (reset - now) // 60
        print('Reset in {} min'.format(minutes))
