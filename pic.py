import requests

body=requests.get("http://gmis.xjtu.edu.cn/pyxx/grxx/xszphd/zp/by/3118311030")
print(body.content)