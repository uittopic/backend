# PHẦN 4: CÀI ĐẶT VÀ TRIỂN KHAI

## 4.1. Cấu trúc Source Code

Hệ thống backend được xây dựng dựa trên framework FastAPI và được tổ chức theo kiến trúc modular để dễ dàng mở rộng và bảo trì trong tương lai. Toàn bộ mã nguồn được chia thành các module rõ ràng, mỗi module đảm nhận một chức năng cụ thể. Thư mục gốc của dự án có tên là `CHUYEN_DE_Backend`, bên trong chứa thư mục `app` là nơi tập trung toàn bộ logic của ứng dụng.

Thư mục `app` được chia thành nhiều thư mục con. Thư mục `api` chứa các file định nghĩa các endpoint của API, trong đó file `routes_caption.py` là nơi xử lý các request liên quan đến việc tạo caption cho ảnh. Thư mục `core` là phần quan trọng nhất, chứa các module cốt lõi của hệ thống. File `config.py` quản lý tất cả các cấu hình của ứng dụng, từ đường dẫn model đến các tham số generation. File `model_loader.py` chịu trách nhiệm load model BLIP đã được fine-tune cho tiếng Việt, còn file `accent_restoration_loader.py` thì load model phục hồi dấu tiếng Việt dựa trên XLM-RoBERTa.

Ngoài ra, thư mục `services` chứa các lớp business logic, giúp tách biệt logic xử lý khỏi phần routing. Thư mục `utils` chứa các hàm tiện ích như cache và rate limiting. File `main.py` ở thư mục `app` là entry point của ứng dụng FastAPI, nơi khởi tạo ứng dụng và đăng ký các middleware như CORS, rate limiting, và authentication nếu cần.

Bên ngoài thư mục `app`, còn có thư mục `configs` chứa các file cấu hình dạng YAML hoặc JSON. Thư mục `models` là nơi lưu trữ các file weights của model đã được train. File `requirements.txt` liệt kê tất cả các thư viện Python cần thiết để chạy dự án.

Về phần code, module `model_loader.py` có nhiệm vụ load model BLIP từ đường dẫn đã được cấu hình. Khi khởi động ứng dụng, module này sẽ kiểm tra xem có model đã được fine-tune chưa. Nếu có, nó sẽ load model đó, còn nếu không thì sẽ fallback về pretrained model từ HuggingFace. Sau khi load xong, model sẽ được chuyển sang thiết bị phù hợp, có thể là MPS trên Mac, CUDA trên máy có GPU NVIDIA, hoặc CPU nếu không có GPU. Việc tự động detect thiết bị giúp hệ thống có thể chạy được trên nhiều môi trường khác nhau mà không cần phải cấu hình thủ công.

Module `routes_caption.py` chứa hàm `_generate_caption_for_image` là hàm xử lý chính để sinh caption từ ảnh. Hàm này đầu tiên sẽ resize ảnh nếu kích thước quá lớn để tối ưu tốc độ xử lý và giảm memory usage, đặc biệt quan trọng khi chạy trên MPS của Mac. Sau đó, ảnh được đưa vào processor để chuyển đổi thành tensor, rồi được đưa vào model để generate caption. Quá trình generation sử dụng beam search với số beams là 3, và có repetition penalty để tránh lặp từ. Kết quả sau khi decode sẽ là caption tiếng Việt không dấu.

Endpoint `/api/caption_full` trong cùng file này là endpoint quan trọng nhất, nhận file ảnh từ client và trả về caption có dấu. Endpoint này thực hiện pipeline hai giai đoạn. Giai đoạn đầu tiên là sinh caption không dấu bằng BLIP model như đã mô tả ở trên. Giai đoạn thứ hai là phục hồi dấu tiếng Việt bằng cách gọi hàm `restore_accent` từ module accent restoration. Kết quả cuối cùng được trả về dưới dạng JSON, bao gồm caption có dấu, caption không dấu, thời gian xử lý, và một số thông tin khác như device đang sử dụng.

## 4.2. Triển khai và Vận hành

Để triển khai hệ thống trên môi trường thực tế, cần thực hiện một số bước chuẩn bị. Đầu tiên là kiểm tra phiên bản Python, hệ thống yêu cầu Python 3.9 trở lên. Sau đó, tạo môi trường ảo bằng lệnh `python3 -m venv .venv` để tránh xung đột với các thư viện khác trên hệ thống. Việc sử dụng virtual environment là một best practice trong Python, giúp quản lý dependencies tốt hơn và đảm bảo tính nhất quán giữa các môi trường khác nhau.

Sau khi tạo xong môi trường ảo, cần kích hoạt nó. Trên MacOS và Linux, dùng lệnh `source .venv/bin/activate`, còn trên Windows thì dùng `.venv\Scripts\activate`. Khi môi trường ảo đã được kích hoạt, prompt của terminal sẽ hiển thị tên môi trường ở đầu dòng, cho biết đang làm việc trong môi trường ảo.

Tiếp theo là cài đặt các thư viện cần thiết. File `requirements.txt` chứa danh sách đầy đủ các thư viện và phiên bản cụ thể. Các thư viện chính bao gồm FastAPI và Uvicorn cho phần web server, Transformers và PyTorch cho phần deep learning, Pillow cho xử lý ảnh, và một số thư viện khác như pandas, numpy cho xử lý dữ liệu. Việc cài đặt được thực hiện bằng lệnh `pip install -r requirements.txt`. Quá trình này có thể mất vài phút tùy thuộc vào tốc độ internet và cấu hình máy tính.

Sau khi cài đặt xong các thư viện, cần đảm bảo rằng các model đã được download và đặt đúng vị trí. Model BLIP đã fine-tune cần được đặt trong thư mục `models/blip_vietnamese_80_20`, còn model accent restoration sẽ được tự động download từ HuggingFace khi lần đầu chạy ứng dụng nếu chưa có sẵn.

Để khởi chạy server, sử dụng Uvicorn làm ASGI server. Uvicorn là một server hiệu năng cao, được thiết kế đặc biệt cho các ứng dụng async như FastAPI. Lệnh khởi chạy là `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`. Tham số `--host 0.0.0.0` cho phép server lắng nghe trên tất cả các interface mạng, không chỉ localhost, điều này cần thiết nếu muốn truy cập từ các máy khác trong cùng mạng. Tham số `--reload` cho phép server tự động reload khi có thay đổi code, rất tiện lợi khi đang phát triển.

Khi server khởi động thành công, sẽ thấy các dòng log hiển thị thông tin về việc load model. Đầu tiên là thông báo đang load BLIP model, sau đó là thông báo đã load xong model fine-tuned tiếng Việt. Tiếp theo là thông báo đang load Accent Restoration model, và cuối cùng là thông báo application startup complete. Nếu thấy các thông báo này, nghĩa là server đã sẵn sàng nhận request.

Hệ thống cung cấp ba endpoint chính. Endpoint `/api/caption` nhận file ảnh và trả về caption tiếng Việt không dấu, chỉ sử dụng BLIP model. Endpoint `/api/caption_full` là endpoint quan trọng nhất, nhận file ảnh và trả về caption tiếng Việt có dấu sau khi đã qua pipeline BLIP và accent restoration. Endpoint `/api/health` dùng để kiểm tra trạng thái của hệ thống, trả về thông tin về việc model đã được load chưa, device đang sử dụng, và một số thông tin khác.

Khi gọi API, client cần gửi request dạng POST với file ảnh trong form-data. Server sẽ đọc file ảnh, chuyển đổi sang định dạng RGB nếu cần, rồi đưa vào pipeline xử lý. Kết quả được trả về dưới dạng JSON, bao gồm trường `success` cho biết request có thành công không, trường `caption_vi` chứa caption có dấu, trường `caption_vi_no_accent` chứa caption không dấu, trường `accent_restored` cho biết có áp dụng accent restoration không, trường `device` cho biết device đang sử dụng, và trường `processing_time` cho biết thời gian xử lý tính bằng giây.

Ví dụ, khi gửi một ảnh áo dài màu hồng, response có thể là một JSON object với `caption_vi` là "một người phụ nữ đang mặc áo dài màu hồng", `caption_vi_no_accent` là "mot nguoi phu nu dang mac ao dai mau hong", `accent_restored` là `true`, `device` là "mps" nếu chạy trên Mac, và `processing_time` có thể là khoảng 0.85 giây tùy thuộc vào cấu hình máy.

Về kiến trúc triển khai, hệ thống được thiết kế theo mô hình đơn giản hóa của microservices. FastAPI đóng vai trò là gateway và controller, nhận request từ client và điều phối việc xử lý. Khi nhận được request, FastAPI sẽ gọi đến preprocessor để chuẩn bị ảnh, sau đó đưa vào BLIP model để sinh caption không dấu. Kết quả từ BLIP được đưa vào accent restoration model để phục hồi dấu, và cuối cùng kết quả được trả về cho client. Các model weights được lưu trữ trong thư mục `models` và được load vào memory khi server khởi động, giúp giảm thời gian xử lý khi có request.

