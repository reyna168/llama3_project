
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd

# 創建圖片保存資料夾
if not os.path.exists('images'):
    os.makedirs('images')


# 多筆
# 目標網站URL
#url = "http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=5"
#url = "http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=6"
#url = "http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=7"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=11"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=13"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=37"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=242"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=15"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=15&page=2"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=274"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=261"

#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=124"

#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=70"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=244"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=245"

#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=245"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=125"
#url ="http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=128"

url = "http://www.yiliang.com.tw/zh-tw/product.php?act=list&cid=49"

web = requests.get(url)
soup = BeautifulSoup(web.text, "html.parser")


products = soup.find_all('h3')

# 儲存產品名稱的列表
productarray = []

for product in products:
    product_name = product.text.strip()  # 取得產品名稱
    productarray.append(product_name)
    print(product_name)  # 印出產品名稱


print("array")

print(productarray)

print("xxxxxx")
# 取得所有img標籤
imgs = soup.find_all('img')

# 儲存圖片URL的列表
image_urls = []

categorys = ["高壓比流器"]*len(imgs)
websiteurl = [url]*len(imgs)


# 下載圖片並儲存到本地
for i in imgs:
    img_url = i.get('data-src') or i.get('src')  # 使用data-src如果有，否則使用src
    if not img_url.startswith('http'):
        img_url = f'http://www.yiliang.com.tw/{img_url}'  # 替換成實際的基礎URL
    print(f'圖片URL: {img_url}')
    
    #categorys*i
    # 將圖片URL加入列表
    image_urls.append(img_url)
    
    # 下載圖片
    img_data = requests.get(img_url).content
    img_name = os.path.join('images', img_url.split('/')[-1])  # 以圖片名儲存
    with open(img_name, 'wb') as handler:
        handler.write(img_data)
    print(f'圖片已下載: {img_name}')

# 將圖片URL保存到Excel
# df = pd.DataFrame(image_urls, columns=["Image URL"])
# df.to_excel('image_urls.xlsx', index=False)
# print("圖片URL已保存至image_urls.xlsx")


import pandas as pd
   
   
# 字典
dict = {'projectname': productarray, 'imageurls': image_urls, "category": categorys , "websiteurl":websiteurl}
     
df = pd.DataFrame(dict)

 
# 保存 dataframe
df.to_csv('web_49.csv')



