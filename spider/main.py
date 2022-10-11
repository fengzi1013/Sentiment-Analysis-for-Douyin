# __author:Administrator
# date: 2022/5/19

# 打开页面后，尽量直接拉到评论下面！！！！



import random
import pandas as pd
import time
from selenium import webdriver
from tqdm import tqdm, trange
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions

# 实现规避检测
option = ChromeOptions()
option.add_experimental_option('excludeSwitches',['enable-automation'])


def getData(url):

    # chromedriver.exe,下载,这个看自己安装的Google的版本，下载解压后放到当前代码路径下。下载地址 http://chromedriver.storage.googleapis.com/index.html
    driver = webdriver.Chrome(options=option)
    driver.get(url)
    time.sleep(20) # 手动点下弹窗关闭登录,或者自己扫码登录！！！

    userNames = []  # 用户名
    userId = []   # 用户抖音号
    userAddress = []   # 用户属地
    userFollow = []   # 用户关注
    userFan = []    # 用户粉丝
    userLiked = []   # 用户获赞数
    timeList = []   # 发表时间
    comments = []   # 评论文本
    likeNums = []   # 该条评论的点赞数


    for i in trange(1, 1000):  # 自行设定爬取条数,不建议太多！！！

        try:
            # 去掉中途出现的登录页面
            driver.find_element(by=By.XPATH,
                                value='//*[@id="login-pannel"]/div[2]').click()
        except:
            try:
                t = random.uniform(1.5, 2)  # 随机浮点数t
                sw = random.randint(150, 180)  # 滑动像素点
                time.sleep(t)  # 睡眠t时间

                # 用户名
                userName = driver.find_element(by=By.XPATH,
                                            value= f"//*[@id='root']//div[{i}]/div/div[2]/div[1]/div[2]/div[1]/div/a/span/span/span/span/span").text

                # 发表时间
                time_ = driver.find_element(by= By.XPATH,
                                            value= f"//*[@id='root']//div[{i}]/div/div[2]/div[1]/div[2]/div[1]/p").text

                # 评论
                comment = driver.find_element(by= By.XPATH,
                                              value= f"//*[@id='root']//div[{i}]/div/div[2]/div[1]/p/span/span/span/span/span/span").text

                # 该条评论的点赞数
                likeNum = driver.find_element(by=By.XPATH,
                                              value= f"//*[@id='root']//div[{i}]/div/div[2]/div[1]/div[3]/div/p/span").text

                # 跳转到用户主页
                # 可能获取不到用户信息，该用户也许注销了 https://www.douyin.com/user/MS4wLjABAAAAOwQt3GN0ydFoV7cEc_bzjS-wT7CWuxOTW7wTcDKS3_c

                driver.find_element(by=By.XPATH,
                                value= f"//*[@id='root']//div[{i}]/div/div[2]/div[1]/div[2]/div[1]/div/a/span/span/span/span/span").click()


                # time.sleep(t)
                driver.switch_to.window(driver.window_handles[1])
                time.sleep(t)

                # 抖音号
                try:
                    id = driver.find_element(by=By.XPATH,
                                             value="//*[@id='root']/div/div[2]/div/div/div[2]/div[1]/p[1]/span[1]").text.split('：')[1].strip()
                except:
                    id = ""
                # 关注
                try:
                    follow = driver.find_element(by=By.XPATH,
                                                 value="//*[@id='root']/div/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div[1]/div[2]").text
                except:
                    follow = ""
                # 粉丝
                try:
                    fan = driver.find_element(by=By.XPATH,
                                              value="//*[@id='root']/div/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div[2]/div[2]").text
                except:
                    fan = ""
                # 获赞
                try:
                    liked = driver.find_element(by=By.XPATH,
                                                value="//*[@id='root']/div/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div[3]/div[2]").text
                except:
                    liked = ""

                # ip属地
                try:
                    ip_address = driver.find_element(by=By.XPATH,
                                         value="//*[@id='root']/div/div[2]/div/div/div[2]/div[1]/p[1]/span[2]").text.split('：')[1].strip()
                except:
                    ip_address = ""

                time.sleep(0.2)
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                driver.execute_script(f"window.scrollBy(0, {sw})")

                userId.append(id)
                userNames.append(userName)
                userAddress.append(ip_address)
                userFan.append(fan)
                userFollow.append(follow)
                userLiked.append(liked)
                timeList.append(time_)
                comments.append(comment)
                likeNums.append(likeNum)
                print(f"第{i}条下载完成！！！")

            except:
                continue

    return userId, userNames, userAddress, userFan, userFollow, userLiked,  timeList, comments, likeNums



if __name__ == "__main__":

    id = "7045926793802501416"  # 这串数字是视频ID
    url = f"https://www.douyin.com/video/{id}"
    userId, userNames, userAddress, userFan, userFollow, userLiked,  timeList, comments, likeNums = getData(url)
    data = pd.DataFrame({"userId":userId, "userName":userNames, "userAddress": userAddress, "userFan": userFan,
                         "userFollow": userFollow, "userLiked": userLiked,"date": timeList, "comments": comments, "likeNuns": likeNums})
    data.to_csv(f"./result_ID{id}.csv") # save path
    print("**********done***********")