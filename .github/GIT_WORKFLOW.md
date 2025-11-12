# 🔀 Git Workflow Guide

Hướng dẫn quy trình làm việc với Git cho dự án **BLIP Vietnamese Captioning API**.

## 📋 Quy tắc chung

### 1. Branch Strategy
- **`main`**: Branch chính, luôn ở trạng thái stable và có thể deploy
- **`feature/*`**: Các tính năng mới (ví dụ: `feature/optimize-vietnamese-caption-generation`)
- **`fix/*`**: Sửa lỗi (ví dụ: `fix/memory-leak`)
- **`docs/*`**: Cập nhật documentation (ví dụ: `docs/update-readme`)

### 2. Commit Message Convention
Sử dụng [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: Tính năng mới
- `fix`: Sửa lỗi
- `docs`: Cập nhật documentation
- `style`: Formatting, thiếu semicolon, etc (không ảnh hưởng code)
- `refactor`: Refactor code
- `test`: Thêm/sửa tests
- `chore`: Cập nhật build tasks, dependencies, etc

**Ví dụ:**
```bash
feat(api): add Vietnamese caption generation endpoint

- Add POST /api/caption endpoint
- Support image upload and URL
- Return JSON with caption text

Closes #123
```

### 3. Pull Request Process

#### Bước 1: Tạo branch mới
```bash
# Từ main branch
git checkout main
git pull origin main

# Tạo branch mới
git checkout -b feature/your-feature-name
```

#### Bước 2: Làm việc và commit
```bash
# Làm thay đổi
# ...

# Add và commit
git add .
git commit -m "feat: your feature description"
```

#### Bước 3: Push và tạo Pull Request
```bash
# Push branch lên remote
git push -u origin feature/your-feature-name
```

Sau đó tạo Pull Request trên GitHub:
- Vào: https://github.com/uittopic/backend/pulls
- Click "New Pull Request"
- Chọn branch của bạn và `main`
- Điền title và description rõ ràng
- Tag reviewer nếu cần

#### Bước 4: Review và Merge
- Đợi review từ đồng nghiệp
- Fix các comments nếu có
- Sau khi approved, merge vào `main`
- Xóa branch sau khi merge (GitHub có option tự động)

## 🚀 Quy trình làm việc hàng ngày

### Khi bắt đầu làm việc
```bash
# 1. Pull latest changes từ main
git checkout main
git pull origin main

# 2. Tạo branch mới cho task của bạn
git checkout -b feature/your-task-name
```

### Trong khi làm việc
```bash
# Commit thường xuyên với message rõ ràng
git add .
git commit -m "feat: add new feature"

# Push để backup code
git push origin feature/your-task-name
```

### Khi hoàn thành task
```bash
# 1. Đảm bảo code đã được push
git push origin feature/your-task-name

# 2. Tạo Pull Request trên GitHub
# 3. Đợi review và merge
```

## 📝 Best Practices

### ✅ Nên làm:
- ✅ Commit thường xuyên với message rõ ràng
- ✅ Tạo branch riêng cho mỗi feature/fix
- ✅ Pull latest changes trước khi tạo branch mới
- ✅ Review code của nhau trước khi merge
- ✅ Test code trước khi push
- ✅ Viết commit message bằng tiếng Anh (chuẩn quốc tế)

### ❌ Không nên:
- ❌ Commit trực tiếp lên `main` branch
- ❌ Commit file nhạy cảm (.env, credentials)
- ❌ Commit file lớn (models, checkpoints) - dùng Git LFS hoặc external storage
- ❌ Force push lên shared branches
- ❌ Commit code chưa test

## 🔒 File không nên commit

Đã được cấu hình trong `.gitignore`:
- Models và checkpoints (`models/`, `*.pt`, `*.pth`)
- Data files (`data/`)
- Environment variables (`.env`)
- Virtual environment (`venv/`, `.venv`)
- Logs (`logs/`, `*.log`)
- IDE configs (`.vscode/`, `.idea/`)

## 🆘 Xử lý conflicts

Khi có conflict khi merge:

```bash
# 1. Pull latest changes
git checkout main
git pull origin main

# 2. Merge main vào branch của bạn
git checkout feature/your-branch
git merge main

# 3. Resolve conflicts trong code editor
# 4. Commit sau khi resolve
git add .
git commit -m "fix: resolve merge conflicts"
git push origin feature/your-branch
```

## 📚 Tài liệu tham khảo

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

## 👥 Cho đồng nghiệp Mobile

Nếu bạn làm việc với mobile app:

1. **Backend API changes**: Khi backend có thay đổi API, sẽ được document trong PR description
2. **Breaking changes**: Sẽ được đánh dấu rõ ràng và có migration guide
3. **API versioning**: Sử dụng version trong URL nếu cần (ví dụ: `/api/v1/caption`)

---

**Lưu ý**: Nếu có thắc mắc về Git workflow, hãy hỏi team trước khi làm để tránh conflicts!

