## Overview
Repo này thuộc môn CS317.P22 (Phát triển và vận hành hệ thống máy học) cho Lab 3 xây dựng Monitoring và Logging service cho API đã được xây dựng trong Lab 2 nằm ở [repo này](https://github.com/Honam0905/CS317.P22)

## Yêu cầu 
- Git ≥ 2.20
- Python 3.9.x
- Docker ≥ 20.10
- Docker Compose plugin (v2)
- (Optional) virtualenv or conda for local Python environment

## Hướng dẫn chạy 
1. Cài Git LFS (chỉ làm 1 lần trên máy)
    Lý do vì file mô hình .pt được up lên bằng việc sử dụng git lfs nên trong đoạn code sẽ gặp vấn đề nếu không dùng command git lfs trước
   ```bash
   git lfs install
   ```
2. Clone repo:
   
   ```bash
   git clone https://github.com/Honam0905/CS317.P22-Lab_3
   cd CS317.P22
   ```
3. Kéo file nhị phân
   ```bash
   git lfs fetch --all
   git lfs checkout
   ```
4. Tạo và kích hoạt virtualenv( Nếu cần thiết)
   
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   ```
5. Cài đặt dependencies
   
   ```bash
   pip install --upgrade pip
   pip install --no-cache-dir -r serve/requirements.txt
   ```
6. Đóng gói & Deploy với Docker<br>
   ```bash
   cd CS317.P22-Lab_3/bert_sentiment_analysis
   ```
   6.1. Ở thư mục gốc, build image:
   
   ```bash
   docker compose build sentiment-api
   ```
   6.2. Khởi chạy container:
   
   ```bash
   docker compose up -d
   ```
   6.3. Kiểm tra trạng thái:
   
   ```bash
   docker ps
   ```
   6.4. Kiểm tra lỗi nếu có:

   ```bash
   docker logs <container-name>/<container-id>
   ```
   6.5  giả lập traffic request
   ```bash
   python scripts/traffic_gen.py
   ```
   * Lưu ý khi chạy xong câu lệnh 7.2 khi bấm vào đường link 8.1 sẽ hiện ra lỗi phải mất một lúc thì mới kết nối được nên khi chạy xong 7.2 chờ đợi một lúc để local host nhận được tín hiệu trừ trường hợp mạng mạnh
8. Gọi API
   - HTML form: http://localhost:8000/  
   - Swagger UI:  http://localhost:8000/docs
   - Prometheus: http://localhost:9090/
   - Grafana: http://localhost:3000/
   - Alertmanager: http://localhost:9093/
   - Node Exporter: http://localhost:9100/
9. Video Demo

<p align="center">
  <a href="https://youtu.be/BaFtFba4yZE?si=yUXGAHsB6GEl7eA5" target="_blank">
    <img
      src="https://img.youtube.com/vi/nDtXXhgrJr8/0.jpg"
      alt="Watch the demo"
      width="600"
    />
  </a>
</p>
