import fitz
from os import listdir
from os.path import isfile, join
# import glob

mypath='..\TCC\images' 
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for file in onlyfiles:
    if file.endswith(".pdf"):
        doc = fitz.open(mypath+'\\'+file)
        page = doc.load_page(0)  # number of page
        pix = page.get_pixmap()
        output = mypath+'\\png\\' +file.strip('.pdf')+".png"
        pix.save(output)
        doc.close()