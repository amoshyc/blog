---
title: "使用 Python 寄發 Gmail"
date: 2018-02-22T17:32:32+08:00
categories: ["Python"]
tags: ["python", "gmail", "smtp", "smtplib", "email"]
toc: true
math: false
---

# 前情提要

之前跟學長在聊天，聊到他需要寄大量通知信給許多人，但希望每封信都有對方的名字。剛好有興趣，就搜了一下網路上的範例，整理出一個使用 python 3.x 從 Gmail 寄發 email 的程式。使用 SMTP、登入並且信件 SSL 加密。修改一下、加個迴圈就可以讓你的 gmail 大量寄垃圾信了（笑~

# 程式碼

{{< highlight python "linenos=table,noclasses=false" >}}
import smtplib
from email.mime.text import MIMEText

gmail_user = 'amoshuangyc@gmail.com'
gmail_password = '---' # your gmail password

msg = MIMEText('content')
msg['Subject'] = 'Test'
msg['From'] = gmail_user
msg['To'] = 'xxx@gmail.com'

server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.ehlo()
server.login(gmail_user, gmail_password)
server.send_message(msg)
server.quit()

print('Email sent!')
{{< / highlight >}}


# 注意事項

1. 第一次寄信時，Google 會寄 email 警告寄信者「查看遭拒的登入嘗試」，開啟「低安全性應用程式」即可。
2. port 465 是 Google [訂定](https://support.google.com/mail/answer/7126229?hl=zh-Hant) 的。
3. Gmail 有單天 500 封及單封 500 人的[限制](https://support.google.com/mail/answer/22839?hl=en)。
4. 你可以不登入直接寄的樣子，只是很可能被判定為垃圾郵件或被 Gmail 說「不是本人寄的」，我沒試過~

