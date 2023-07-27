from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

import time

#configure selenium
chromedriver = "D:\Frontend testing\chromedriver"
driver = webdriver.Chrome(chromedriver)

#open nipgboard
driver.get("https://www.google.com")



# #go to registration
# register_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[1]/form/paper-button[2]")
# register_button.click()

# tc = unittest.TestCase()

# #get the input elements
# name = driver.find_element_by_id("input-14")
# password = driver.find_element_by_id("input-15")
# global_password = driver.find_element_by_id("input-16")
# folder_name = driver.find_element_by_id("input-17")

# time.sleep(0.2)
# '''
# ---WRONG USERNAME---
# '''
# name.send_keys("teszt") #wrong username -> too short
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "The username is too short! Minimum length is 6 character!")


# '''
# ---WRONG PASSWORD--- 
# 1. Password - short
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Teszt2") #-> wrong password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "The password is too short! Minimum length is 6 character!")


# '''
# 2. Password - no lowercase
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("TESZTPASSWORD2021") #-> wrong password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "The password should contain lowercase character!")


# '''
# 3. Password - no uppercase
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("tesztpassword2021") #-> wrong password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "The password should contain uppercase character!")

# '''
# 4. Password - no digit
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Tesztpassword") #-> wrong password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "The password should contain a digit!")

# '''
# 5.Global Password - incorrect
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global20") #-> wrong global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Incorrect global password!")


# '''
# 6.Global Password - incorrect
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("Global200") #-> wrong global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Incorrect global password!")


# '''
# 7.Global Password - incorrect
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("lobal200") #-> wrong global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Incorrect global password!")

# '''
# 9.Foldername short
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("Tesztusername2021") #correct username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("teszt") #->wrong sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "This foldername is too short! Minimum length is 6 character!")


# '''
# 10.Correct Registration
# '''
# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #correct username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global200") #-> wrong global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Successful registration!")

# #back to registration
# register_button.click()


# '''
# 11.Taken username
# '''
# time.sleep(0.2)

# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername") #taken username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global200") #-> correct global password
# folder_name.send_keys("tesztlogdir2") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "This username is already taken!")


# '''
# 12.Taken username
# '''

# name.clear()
# password.clear()
# global_password.clear()
# folder_name.clear()

# name.send_keys("tesztusername2") #taken username
# password.send_keys("Tesztpassword2021") #-> correct password
# global_password.send_keys("global200") #-> wrong global password
# folder_name.send_keys("tesztlogdir") #->correct sublogdir

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[2]")
# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-11")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "This foldername is already taken!")


# #back to login
# back_btn = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[2]/form/paper-button[1]")
# back_btn.click()

# name = driver.find_element_by_id("input-12")
# password = driver.find_element_by_id("input-13")

# submit_button = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/div[2]/div[1]/form/paper-button[1]")

# '''
# 13. Login
# '''

# #tesztusername"
# #Tesztpassword2021"

# name.clear()
# password.clear()

# name.send_keys("tesZtusername") #wrong username
# password.send_keys("Tesztpassword2021") #-> correct password

# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-10")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Incorrect username or password, try again!")


# '''
# 14. Login
# '''

# #tesztusername"
# #Tesztpassword2021"

# name.clear()
# password.clear()

# name.send_keys("tesztusername") #wrong username
# password.send_keys("tesztpassword2021") #-> correct password

# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-10")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Incorrect username or password, try again!")

# '''
# 15. Login
# '''

# #tesztusername"
# #Tesztpassword2021"

# name.clear()
# password.clear()

# name.send_keys("tesztusername") #wrong username
# password.send_keys("Tesztpassword2021") #-> correct password

# submit_button.click()

# time.sleep(0.2)

# notification = driver.find_element_by_id("input-10")
# text = driver.execute_script("return arguments[0].placeholder", notification)

# text = text.split('\n')
# tc.assertEqual(text[-2], "Correct!")

# time.sleep(3)

# removepairs_button = driver.find_element_by_id("removeAll")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", submit_button)
# tc.assertFalse(isdisabled)

# #execute button
# execute_btn = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[1]/vz-multidash-dashboard/div[1]/div[1]/div[2]/vz-executer-dashboard/vz-executer/div/div[3]/paper-button")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", execute_btn)
# tc.assertFalse(isdisabled)

# #download
# download_btn = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[1]/vz-multidash-dashboard/div[1]/div[4]/div[2]/vz-projector-dashboard/vz-projector/div/div[1]/vz-projector-data-panel/div[2]/div[4]/span[3]/paper-button")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", download_btn)
# tc.assertFalse(isdisabled)

# share = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-icon-button/iron-icon")
# share.click()

# time.sleep(0.2)

# dis1 = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[1]/paper-checkbox[1]/div[1]")
# dis1.click()

# time.sleep(0.2)


# dis2 = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[1]/paper-checkbox[2]/div[1]")
# dis2.click()

# time.sleep(0.2)

# dis3 = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[1]/paper-checkbox[3]/div[1]")
# dis3.click()

# time.sleep(0.2)

# dis4 = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[1]/paper-checkbox[4]/div[1]")
# dis4.click()

# time.sleep(0.2)

# dis5 = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[1]/paper-checkbox[5]/div[1]")
# dis5.click()

# time.sleep(0.2)

# copy = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[4]/vz-multidash-dashboard/paper-dialog/div[1]/div[2]/input")
# time.sleep(1)
# sharelink = driver.execute_script("return arguments[0].value", copy)

# driver.get('https://www.google.com/')

# time.sleep(1)

# driver.find_element_by_xpath('/html/body').send_keys(Keys.COMMAND + 't')
# driver.get(sharelink)

# time.sleep(3)

# removepairs_button = driver.find_element_by_id("removeAll")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", removepairs_button)
# tc.assertTrue(isdisabled)

# #execute button
# execute_btn = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[1]/vz-multidash-dashboard/div[1]/div[1]/div[2]/vz-executer-dashboard/vz-executer/div/div[3]/paper-button")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", execute_btn)
# tc.assertTrue(isdisabled)

# #download
# download_btn = driver.find_element_by_xpath("/html/body/tf-tensorboard/paper-header-panel/div/div[1]/div/div/div[1]/vz-multidash-dashboard/div[1]/div[4]/div[2]/vz-projector-dashboard/vz-projector/div/div[1]/vz-projector-data-panel/div[2]/div[4]/span[3]/paper-button")
# isdisabled = driver.execute_script("return arguments[0].hasAttribute(\"disabled\");", download_btn)
# tc.assertTrue(isdisabled)

# print('OK')