# 🔧 Hướng dẫn cấu hình IDE

## Lỗi: "Import transformers could not be resolved"

Đây là lỗi IDE không nhận diện Python interpreter đúng. Làm theo các bước sau:

### ✅ Giải pháp 1: VS Code

1. Mở Command Palette: `Cmd + Shift + P` (macOS) hoặc `Ctrl + Shift + P` (Windows/Linux)
2. Gõ: `Python: Select Interpreter`
3. Chọn: `./venv/bin/python` (virtual environment của project)

Hoặc file `.vscode/settings.json` đã được tạo sẵn, VS Code sẽ tự động nhận diện.

### ✅ Giải pháp 2: PyCharm

1. Vào `PyCharm` → `Preferences` → `Project` → `Python Interpreter`
2. Click `⚙️` → `Add...`
3. Chọn `Existing environment`
4. Trỏ đến: `venv/bin/python`

### ✅ Giải pháp 3: Kiểm tra lại

```bash
# Kích hoạt virtual environment
source venv/bin/activate

# Kiểm tra transformers đã cài chưa
python -c "import transformers; print('✅ OK')"
```

### 📝 Lưu ý

- Luôn kích hoạt virtual environment trước khi chạy code:
  ```bash
  source venv/bin/activate
  ```

- Nếu vẫn lỗi, restart IDE sau khi chọn interpreter.

