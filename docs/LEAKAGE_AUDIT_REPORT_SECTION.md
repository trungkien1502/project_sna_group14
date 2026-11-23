Kiểm tra bổ sung: tỷ lệ nhắc tên sách theo từng tập

Kiểm tra trùng tên sách ban đầu ghi nhận 29 lượt trùng với sách validation/test (13 validation, 16 test), thuộc 24 user. Để xác định các trường hợp này có phải do rò rỉ dữ liệu hay không, nhóm so sánh tỷ lệ user profile nhắc tên sách giữa các tập. Nếu sách validation/test được đưa vào prompt, LLM không phân biệt được chúng với sách train, nên sách validation/test phải được nhắc tên với tỷ lệ gần bằng sách train.

Cách thực hiện: Với mỗi cặp user–sách, nhóm kiểm tra title của sách (đã chuẩn hóa, tối thiểu 2 từ) có xuất hiện trong profile hoặc reasoning của user đó không. Nhóm đối chứng gồm 3 sách user chưa từng tương tác, chọn ngẫu nhiên trong cùng nhóm độ phổ biến. Quy tắc này lỏng hơn bước kiểm tra ban đầu nên ghi nhận nhiều lượt trùng hơn (62 so với 29).

Tập
	Số cặp kiểm tra
	Số lượt nhắc tên
	Tỷ lệ nhắc
	Nhóm đối chứng
	Train
	108.525
	15.450
	14,24%
	0,005%
	Validation
	36.354
	25
	0,069%
	0,005%
	Test
	36.001
	37
	0,103%
	0,001%
	Đọc kết quả: Sách train được nhắc tên ở 14,24% cặp, trong khi sách validation/test chỉ khoảng 0,1%, thấp hơn khoảng 150 lần. Nếu lịch sử validation/test có trong prompt, kỳ vọng có khoảng 10.300 lượt nhắc; thực tế chỉ có 62. Theo khoảng tin cậy 95%, phần lịch sử validation/test có thể đã bị đưa vào prompt tối đa khoảng 0,8%.

Đối chiếu 62 lượt trùng với lịch sử train của từng user:
• Sách cùng series có trong train (56 lượt): ví dụ user có "Catching Fire (The Hunger Games, Book 2)" trong train, profile nhắc "The Hunger Games trilogy" nên trùng với title cuốn 1 trong test.
• Cùng một cuốn sách nhưng khác ASIN có trong train (2 lượt): item 393 trong train không có title trong metadata, nhưng item profile của tác giả ghi là "Present Perfect by Alison G. Bailey", tức cùng sách với item 856 trong validation.
• Kiến thức sẵn có của LLM hoặc cụm từ phổ biến (4 lượt): ví dụ profile gợi ý "The Black Swan" của Nassim Taleb cho user đã đọc "Fooled by Randomness" cùng tác giả.
Không có lượt trùng nào cần đến giả định rò rỉ dữ liệu để giải thích.

Kết luận: Nhóm không tìm thấy bằng chứng user profiles được tạo từ lịch sử validation/test, nên giữ nguyên semantic profiles và embeddings của RLMRec cho đánh giá chính; kết quả E1–E4 không cần chạy lại. Giới hạn: Kiểm tra dựa trên việc profile có nhắc tên sách hay không. Trường hợp thông tin validation/test ảnh hưởng gián tiếp đến profile mà không nêu tên sách không được phát hiện trực tiếp. Paper không nêu rõ profiles được tạo chỉ từ dữ liệu train; mục 3.2.3 chỉ ghi user profile được tạo từ một tập con ngẫu nhiên các sách user đã tương tác. Ngoài ra, theo mục 3.2.2, item profile dùng title và description, và khi thiếu description thì dùng một số review lấy ngẫu nhiên; kiểm tra này chưa xác định được các review đó có thuộc validation/test hay không.

Dữ liệu: usr_prf.pkl, trn_mat.pkl, val_mat.pkl, tst_mat.pkl và data/manifests/amazon_book_titles.json trên Google Drive; mã kiểm tra src/audit/profile_leakage_forensic.py.
