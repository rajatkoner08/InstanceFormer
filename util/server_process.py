# ------------------------------------------------------------------------
# Server evaluation code for InstanceFormer
# ------------------------------------------------------------------------

import glob
import os
import time
from playwright.sync_api import sync_playwright

competition_id = {'youtubeVIS19': '6064',
                  'youtubeVIS21': '7680',
                  'youtubeVIS22': '3410',
                  'OVIS': '32377'}

def upload_file(file, competition='youtubeVIS19'):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context()
        # Open new page
        page = context.new_page()
        page.set_default_timeout(0)

        # Sign-in
        page.goto("https://codalab.lisn.upsaclay.fr/accounts/login/")
        page.locator("#id_login").fill("your_user_name")
        page.locator("#id_password").fill("password")
        page.locator("form.login button[type=submit]").click()
        # Upload zip
        page.goto(f"https://codalab.lisn.upsaclay.fr/competitions/{competition_id[competition]}#participate-submit_results")

        page.locator("#s3_upload_form input[type=file]").set_input_files(file)
        page.locator("#s3-file-upload").click()
        # Close page
        context.close()
        browser.close()


if __name__ == '__main__':
    # Upload all the checkpoints of one Run
    path = 'your_root/instanceformer_output/results/r50_ovis/'
    list_of_files = sorted(filter(os.path.isfile, glob.glob(path + '*.zip')), reverse=True)
    for file in list_of_files:
        print(file)
        upload_file(file, competition='OVIS')
        time.sleep(60)

