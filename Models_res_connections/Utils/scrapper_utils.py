import pickle

# Import Necessary Libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

from Utils.PickleIO import *


# Login to Artsy
def artsy_login(driver, email, password):
    header = driver.find_element(By.TAG_NAME, "header")
    login_button = [i for i in header.find_elements(By.TAG_NAME, "button") if i.text == 'Log In']

    if len(login_button) == 1:
        login_button = login_button[0]
        login_button.click()
        time.sleep(2)

        email_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[placeholder="Enter your email address"]'))
        )
        email_input.send_keys(email)
        time.sleep(2)

        continue_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
        )
        continue_button.click()
        time.sleep(2)

        password_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[placeholder="Enter your password"]'))
        )
        password_input.send_keys(password)
        time.sleep(2)

        login_submit = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
        )
        login_submit.click()
        time.sleep(2)

        # if authentication code is required
        if driver.find_elements(By.CSS_SELECTOR, '[placeholder="Enter an authentication code"]'):
            # prompt for input
            authentication_code = input("Please enter the authentication code received: ")
            driver.find_element(By.CSS_SELECTOR, '[placeholder="Enter an authentication code"]').send_keys(authentication_code)
            time.sleep(1)

            authentication_code_submit = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[@type="submit"]'))
            )
            authentication_code_submit.click()
            time.sleep(2)

        print("Logged in")
        time.sleep(8)
    else:
        print("Error: Not able to locate log in button")