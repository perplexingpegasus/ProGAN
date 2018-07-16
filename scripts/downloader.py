import os
import requests

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


subreddit = input('Enter subreddit name: ')
save_dir = input('Enter name of folder to save images in: ')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

pages = 100
img_n = 0
browser = webdriver.Firefox()
browser.get('https://old.reddit.com/r/{}'.format(subreddit))

for i in range(pages):
    icons = WebDriverWait(browser, 300).until(
        EC.presence_of_all_elements_located(
            (By.CLASS_NAME, "expando-button")
        )
    )

    for icon in icons:
        icon.click()

    links = WebDriverWait(browser, 300).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "may-blank"))
    )
    links = list(set([a.get_attribute('href') for a in links if a.get_attribute('href').endswith('.jpg')]))

    for link in links:
        image = requests.get(link)
        with open('{}/img_{}.jpg'.format(save_dir, img_n), 'wb') as f:
            f.write(image.content)
        img_n += 1

    if i != pages - 1:
        next_button = WebDriverWait(browser, 300).until(
            EC.presence_of_element_located((By.CLASS_NAME, "next-button"))
        )
        next_button.click()

    print('page: {}, images: {}'.format(i, len(links)))