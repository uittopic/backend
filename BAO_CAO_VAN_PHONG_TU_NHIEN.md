# BÁO CÁO MẪU - VĂN PHONG TỰ NHIÊN

## Hướng dẫn sử dụng

File này chứa các đoạn văn mẫu với văn phong tự nhiên, không AI, để bạn tham khảo khi viết lại báo cáo. Các đoạn văn này:
- Sử dụng câu văn có liên kết, không quá nhiều bullet list
- Có một số lặp lại nhẹ để tự nhiên
- Giọng văn người Việt bình thường
- Giải thích theo câu chữ, không "quá sạch sẽ"

---

## PHẦN 1: TỔNG QUAN (Mẫu)

Việc tạo caption tự động cho ảnh sản phẩm là một bài toán quan trọng trong thương mại điện tử hiện nay. Khi người dùng tìm kiếm sản phẩm, việc có caption mô tả chính xác giúp cải thiện đáng kể trải nghiệm tìm kiếm và tăng khả năng người dùng tìm thấy đúng sản phẩm mình cần. Tuy nhiên, việc tạo caption thủ công cho hàng nghìn sản phẩm là không khả thi, đặc biệt là với các sàn thương mại điện tử lớn như Shopee hay Lazada.

Bài toán này trở nên phức tạp hơn khi làm việc với tiếng Việt. Khác với tiếng Anh, tiếng Việt có hệ thống dấu thanh phức tạp, và việc sinh caption có dấu chính xác là một thách thức lớn. Các model caption hiện tại thường được train trên dữ liệu tiếng Anh, và khi áp dụng trực tiếp cho tiếng Việt thì kết quả thường không tốt, đặc biệt là phần dấu thanh.

Trong nghiên cứu này, chúng tôi đề xuất một pipeline hai giai đoạn để giải quyết bài toán tạo caption tiếng Việt có dấu cho ảnh sản phẩm. Giai đoạn đầu tiên sử dụng model BLIP đã được fine-tune trên dữ liệu tiếng Việt để sinh caption không dấu. Giai đoạn thứ hai sử dụng model XLM-RoBERTa để phục hồi dấu thanh cho caption đã được sinh ra. Cách tiếp cận này cho phép tận dụng được sức mạnh của các model caption hiện đại trong khi vẫn đảm bảo được độ chính xác về mặt ngữ pháp và chính tả tiếng Việt.

---

## PHẦN 2: CƠ SỞ LÝ THUYẾT (Mẫu)

Model BLIP là một trong những model caption hiện đại nhất hiện nay, được phát triển bởi Salesforce Research. BLIP là viết tắt của Bootstrapping Language-Image Pre-training, một phương pháp pre-training mới giúp model học được cả khả năng hiểu ảnh và khả năng sinh text mô tả ảnh một cách hiệu quả. Điểm mạnh của BLIP là kiến trúc unified, cho phép model vừa có thể làm nhiệm vụ understanding vừa có thể làm nhiệm vụ generation, không như các model trước đó thường chỉ làm được một trong hai nhiệm vụ này.

Kiến trúc của BLIP bao gồm một vision encoder dựa trên Vision Transformer để encode ảnh thành các feature vectors, và một text decoder dựa trên BERT để sinh ra caption. Hai phần này được kết nối với nhau thông qua một cross-attention mechanism, cho phép text decoder có thể "nhìn" vào các phần của ảnh khi sinh từng từ trong caption. Cơ chế này giúp model có thể tạo ra caption chính xác và chi tiết hơn so với các phương pháp truyền thống.

Tuy nhiên, BLIP được train chủ yếu trên dữ liệu tiếng Anh, nên khi áp dụng trực tiếp cho tiếng Việt thì kết quả không tốt. Để giải quyết vấn đề này, chúng tôi đã fine-tune BLIP trên một dataset tiếng Việt gồm hàng nghìn cặp ảnh-caption. Quá trình fine-tune giúp model học được các đặc thù của tiếng Việt, như cách mô tả sản phẩm, cách sử dụng từ ngữ phù hợp với ngữ cảnh thương mại điện tử.

Sau khi fine-tune, BLIP có thể sinh được caption tiếng Việt, nhưng caption này thường không có dấu thanh. Điều này xảy ra vì trong quá trình training, model học được cách sinh từ tiếng Việt nhưng không học được cách đặt dấu thanh chính xác. Để giải quyết vấn đề này, chúng tôi sử dụng thêm một model accent restoration dựa trên XLM-RoBERTa.

XLM-RoBERTa là một biến thể của RoBERTa được train trên nhiều ngôn ngữ, trong đó có tiếng Việt. Model accent restoration mà chúng tôi sử dụng được train để nhận vào một câu tiếng Việt không dấu và trả về câu tiếng Việt có dấu tương ứng. Model này hoạt động ở mức token classification, tức là với mỗi token trong câu không dấu, model sẽ phân loại xem token đó nên có dấu gì. Cách tiếp cận này cho kết quả tốt và nhanh hơn so với các phương pháp khác như sequence-to-sequence.

---

## PHẦN 3: PHÂN TÍCH & THIẾT KẾ HỆ THỐNG (Mẫu)

Hệ thống được thiết kế theo kiến trúc pipeline hai giai đoạn như đã mô tả ở phần tổng quan. Giai đoạn đầu tiên sử dụng BLIP model đã được fine-tune để sinh caption không dấu từ ảnh đầu vào. Giai đoạn thứ hai sử dụng accent restoration model để phục hồi dấu thanh cho caption đã được sinh ra.

Ở giai đoạn đầu tiên, khi nhận được ảnh đầu vào, hệ thống sẽ resize ảnh về kích thước phù hợp nếu ảnh quá lớn. Việc này giúp giảm memory usage và tăng tốc độ xử lý, đặc biệt quan trọng khi chạy trên các thiết bị có GPU yếu hoặc khi cần xử lý nhiều ảnh cùng lúc. Sau đó, ảnh được đưa vào BLIP processor để chuyển đổi thành tensor, rồi được đưa vào BLIP model để generate caption. Quá trình generation sử dụng beam search với số beams là 3, giúp tìm được caption tốt hơn so với greedy decoding. Ngoài ra, hệ thống còn sử dụng repetition penalty để tránh việc model lặp lại cùng một từ nhiều lần, một vấn đề thường gặp trong các model generation.

Caption sau khi được sinh ra sẽ là một câu tiếng Việt không dấu. Câu này được đưa vào giai đoạn thứ hai, tức là accent restoration. Ở giai đoạn này, câu không dấu được tokenize thành các tokens, sau đó được đưa vào XLM-RoBERTa model để phân loại dấu thanh cho từng token. Kết quả sau khi được merge lại sẽ là câu tiếng Việt có dấu hoàn chỉnh.

Toàn bộ pipeline này được tích hợp vào một API RESTful sử dụng FastAPI, cho phép các ứng dụng khác có thể gọi và sử dụng dịch vụ một cách dễ dàng. API được thiết kế để có thể xử lý cả single image và batch images, giúp tăng hiệu quả khi cần xử lý nhiều ảnh cùng lúc. Ngoài ra, hệ thống còn có cơ chế caching để tránh việc phải xử lý lại những ảnh đã được xử lý trước đó, giúp giảm tải cho server và tăng tốc độ response.

---

## PHẦN 4: CÀI ĐẶT VÀ TRIỂN KHAI

(Xem file `BAO_CAO_PHAN_4_MAU.md` đã tạo ở trên)

---

## PHẦN 5: ĐÁNH GIÁ VÀ KẾT QUẢ (Mẫu)

Để đánh giá hiệu quả của hệ thống, chúng tôi đã thực hiện đánh giá trên tập test gồm 1528 ảnh sản phẩm. Tập test này được tách ra từ dataset gốc theo tỷ lệ 80-20, tức là 80% dữ liệu dùng cho training và 20% dùng cho testing. Việc tách tập test này được thực hiện một cách ngẫu nhiên để đảm bảo tính đại diện của dữ liệu.

Chúng tôi sử dụng ba metric chính để đánh giá. Metric đầu tiên là BLEU score, một metric phổ biến trong các bài toán machine translation và image captioning. BLEU score đo độ tương đồng giữa caption được sinh ra và caption tham chiếu dựa trên n-gram overlap. Tuy nhiên, kết quả BLEU score của hệ thống chỉ đạt 0.0141, một con số khá thấp. Sau khi phân tích, chúng tôi nhận thấy nguyên nhân chính là do caption tham chiếu thường rất dài và chi tiết, trong khi caption được sinh ra thường ngắn gọn hơn. Ngoài ra, một số caption được sinh ra có hiện tượng lặp từ, làm giảm điểm BLEU.

Metric thứ hai là ROUGE-L, một metric đo độ tương đồng dựa trên longest common subsequence. ROUGE-L thường phù hợp hơn với các bài toán summarization và captioning vì nó không quá nghiêm khắc về thứ tự từ như BLEU. Kết quả ROUGE-L của hệ thống là 0.1486, một con số ở mức trung bình. Con số này cho thấy hệ thống có thể sinh được các caption có một số điểm chung với caption tham chiếu, nhưng vẫn còn nhiều chỗ cần cải thiện.

Metric thứ ba là SBERT similarity, một metric đo độ tương đồng ngữ nghĩa giữa hai câu bằng cách sử dụng Sentence-BERT để encode cả hai câu thành các vector, rồi tính cosine similarity giữa hai vector này. Kết quả SBERT similarity của hệ thống là 0.6330, một con số khá tốt. Con số này cho thấy mặc dù caption được sinh ra có thể khác về mặt từ ngữ so với caption tham chiếu, nhưng về mặt ngữ nghĩa thì chúng khá tương đồng. Điều này có nghĩa là hệ thống đã hiểu được nội dung của ảnh và có thể mô tả đúng những gì trong ảnh, chỉ là cách diễn đạt có thể khác một chút.

Sau khi phân tích kỹ hơn, chúng tôi nhận thấy rằng việc BLEU score thấp nhưng SBERT similarity cao là một hiện tượng khá phổ biến trong các bài toán captioning. Nguyên nhân là vì có nhiều cách khác nhau để mô tả cùng một ảnh, và các cách mô tả này có thể khác nhau về từ ngữ nhưng vẫn giống nhau về mặt ngữ nghĩa. Ví dụ, một ảnh có thể được mô tả là "áo dài màu hồng" hoặc "chiếc áo dài có màu hồng", hai cách mô tả này khác nhau về từ ngữ nhưng về mặt ngữ nghĩa thì giống nhau. SBERT similarity có thể nắm bắt được điều này, trong khi BLEU thì không.

Về phần accent restoration, hệ thống hoạt động khá tốt. Hầu hết các từ được phục hồi dấu chính xác, chỉ có một số trường hợp ngoại lệ với các từ ít gặp hoặc các từ có nhiều cách đọc khác nhau. Tuy nhiên, những trường hợp này không ảnh hưởng nhiều đến chất lượng tổng thể của caption.

---

## PHẦN 6: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN (Mẫu)

Trong nghiên cứu này, chúng tôi đã xây dựng thành công một hệ thống tạo caption tiếng Việt có dấu cho ảnh sản phẩm sử dụng pipeline hai giai đoạn. Hệ thống sử dụng BLIP model đã được fine-tune trên dữ liệu tiếng Việt để sinh caption không dấu, sau đó sử dụng accent restoration model để phục hồi dấu thanh. Kết quả đánh giá cho thấy hệ thống có thể hiểu được nội dung của ảnh và sinh ra caption có ý nghĩa, mặc dù vẫn còn một số hạn chế về độ chính xác từ ngữ.

Một trong những điểm mạnh của hệ thống là khả năng tích hợp dễ dàng vào các ứng dụng khác thông qua API RESTful. Điều này giúp các nhà phát triển có thể sử dụng dịch vụ một cách thuận tiện mà không cần phải hiểu sâu về các model bên trong. Ngoài ra, hệ thống còn có cơ chế caching và batch processing, giúp tăng hiệu quả khi cần xử lý nhiều ảnh cùng lúc.

Tuy nhiên, hệ thống vẫn còn một số hạn chế. Đầu tiên là vấn đề repetition, một số caption được sinh ra có hiện tượng lặp lại cùng một từ nhiều lần. Vấn đề này có thể được giải quyết bằng cách điều chỉnh các tham số generation như repetition penalty hoặc no-repeat n-gram size. Thứ hai là độ dài của caption, một số caption được sinh ra quá ngắn so với caption tham chiếu. Vấn đề này có thể được giải quyết bằng cách tăng max_new_tokens hoặc điều chỉnh length penalty.

Về hướng phát triển trong tương lai, có một số hướng có thể được theo đuổi. Đầu tiên là cải thiện chất lượng của BLIP model bằng cách train trên dataset lớn hơn hoặc sử dụng các kỹ thuật data augmentation. Thứ hai là cải thiện accent restoration model bằng cách train trên dataset chuyên biệt cho domain thương mại điện tử. Thứ ba là tích hợp thêm các tính năng như multi-modal search, cho phép người dùng tìm kiếm sản phẩm bằng cả text và ảnh. Cuối cùng là tối ưu hóa hiệu năng của hệ thống để có thể xử lý được nhiều request hơn trong cùng một khoảng thời gian.

