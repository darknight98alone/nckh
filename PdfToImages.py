from sys import argv
from pdf2image import convert_from_path
# file,pdf_file,output_folder = argv
def pdfToImage(pdf_file,output_folder):
        print("start")
        i=0
        from PyPDF2 import PdfFileReader
        pdf = PdfFileReader(open(pdf_file,'rb'))
        maxPages = pdf.getNumPages()
        # if maxPages >2:
                # maxPages = 2
        #  = convert_from_path(pdf_file, 500)
        for page in range(1,maxPages,10) : 
                images_from_path = convert_from_path(pdf_file, dpi=200, first_page=page, last_page = min(page+10-1,maxPages))
                for image in images_from_path:
                        image.save(output_folder+str(i)+'.jpg',)
                        i+=1
        