# Code from https://www.256kilobytes.com/content/show/4903/downloading-bulk-images-thispersondoesnotexist-with-python-and-urllib2

## Download images from thispersondoesnotexit.com

import urllib.request

import time
from PIL import Image

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import csv
import base64

# The directory to put the downloaded files into
d = "./TpDne_images/"

# Create the directory, if it doesn't already exist
if not os.path.exists(d):
    os.makedirs(d)

# The url that images are shown at
u = "https://thispersondoesnotexist.com/image"

# The user agent. Replace this with something descriptive to your use case, probably.
ua = "Some bot: https://www.256kilobytes.com/content/show/4903/"

# The HTTP_REFERER metadata in the request header. Optionally replace with your own URL.
r = "https://www.256kilobytes.com/content/show/4903/"

for x in range(0, 100000):
    print("")
    print("// ===== ===== File " + str(x) + " ===== ===== //")
    print("Downloading content from url: " + u);

    req = urllib.request.Request(u)
    req.add_header('Referer', r)
    req.add_header('User-Agent', ua)

    opener = urllib.request.build_opener()
    resp = opener.open(req)
    content = resp.read()

    fn = d + 'TpDne' + str(x).zfill(5) + '.jpeg'
    print("Writing to " + fn + "...")
    f = open(fn, "wb")
    f.write(content)

    # Wait some time between requests to not DDoS/make excessive requests to the server
    # Also, if this time is too short, the page might not refresh and duplicate images will be downloaded
    time.sleep(1)
    print
    ""