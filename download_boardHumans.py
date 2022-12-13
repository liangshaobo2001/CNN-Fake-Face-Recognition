## Download images from boredhumans

import requests

for i in range(10000):
    # image_url = 'https://boredhumans.b-cdn.net/faces2/' + str(i+1) + '.jpg'  # Good (new) fake faces.
    image_url = 'https://boredhumans.b-cdn.net/faces/' + str(i+1) + '.jpg'  # Bad (old) fake faces.
    img_data = requests.get(image_url).content
    img_name = './BH_Bad_images/' + str(i) + '.jpg'
    with open(img_name, 'wb') as handler:
        handler.write(img_data)
    handler.close()
    print(i)
