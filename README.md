# file detai.py
- need install tesseract (build from sources v5.0),pytesseract, imutils python-docx pdf2Image 
- for run:
python3 detai.py -i /home/phamvandan/Documents/research/data/nckh/12 -o /home/phamvandan/Documents/research/code/learn -n output.docx

- tham số sau i là đường dẫn tuyệt đối tới thư mục chứa ảnh
- tham số sau o là đường dẫn sẽ lưu kết quả
- tham số sau n là tên file docx sẽ lưu kết quả

- Chỉ sử dụng 2 file detai.py và skew.py là được

- hiện tại đã:
+ xử lí đầu vào ảnh, pdf, ảnh nghiêng, ảnh có bảng
+ xuất kết quả ra file word
+ chưa xử lí layout
