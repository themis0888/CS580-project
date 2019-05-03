import urllib.request

url = "https://www.dropbox.com/sh/jh03upspuc4ghpd/AADmu_DW8YJpJR0meid28UQSa/README.md?dl=1"
u = urllib.request.urlopen(url)
data = u.read()
u.close()
 
with open("download_test", "wb") as f :
    f.write(data)