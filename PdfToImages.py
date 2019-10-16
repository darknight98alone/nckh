from sys import argv
from pdf2image import convert_from_path
from PyPDF2 import PdfFileReader
def pdfToImage(pdf_file,output_folder):
        print("start")
        i=1
        pdf = PdfFileReader(open(pdf_file,'rb'))
        maxPages = pdf.getNumPages()
        for page in range(1,maxPages+1,10) : 
                images_from_path = convert_from_path(pdf_file, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))
                for image in images_from_path:
                        image.save(output_folder+str(i)+'.jpg',)
                        i = i + 1