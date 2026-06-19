# Thiết lập GitHub

Máy hiện tại đã có local Git repo và các branch:

```text
main
develop
app
Quang
Hung
Khoa
```

## Cách tạo repo trên GitHub

1. Vào GitHub và tạo repo mới.
2. Tên repo đề xuất:

```text
oil-spill-sar-segmentation
```

3. Không chọn tạo sẵn `README`, `.gitignore` hoặc license vì local repo đã có các file này.
4. Sau khi tạo repo, copy URL dạng HTTPS, ví dụ:

```text
https://github.com/<username>/oil-spill-sar-segmentation.git
```

## Kết nối local repo với GitHub

Thay `<repo-url>` bằng URL GitHub vừa tạo:

```bash
git remote add origin <repo-url>
git push -u origin main
git push -u origin develop
git push -u origin app
git push -u origin Quang
git push -u origin Hung
git push -u origin Khoa
```

## Nếu đã cài GitHub CLI

Có thể tạo repo và push bằng:

```bash
gh repo create oil-spill-sar-segmentation --private --source=. --remote=origin --push
git push -u origin develop
git push -u origin app
git push -u origin Quang
git push -u origin Hung
git push -u origin Khoa
```

Nếu muốn repo public, đổi `--private` thành `--public`.

