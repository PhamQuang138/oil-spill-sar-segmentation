# GitHub Setup

May hien tai da co local Git repo va cac branch:

```text
main
develop
app
Quang
Hung
Khoa
```

## Cach tao repo tren GitHub

1. Vao GitHub va tao repo moi.
2. Ten repo de xuat:

```text
oil-spill-sar-segmentation
```

3. Khong chon tao san `README`, `.gitignore` hoac license vi local repo da co cac file nay.
4. Sau khi tao repo, copy URL dang HTTPS, vi du:

```text
https://github.com/<username>/oil-spill-sar-segmentation.git
```

## Ket noi local repo voi GitHub

Thay `<repo-url>` bang URL GitHub vua tao:

```bash
git remote add origin <repo-url>
git push -u origin main
git push -u origin develop
git push -u origin app
git push -u origin Quang
git push -u origin Hung
git push -u origin Khoa
```

## Neu da cai GitHub CLI

Co the tao repo va push bang:

```bash
gh repo create oil-spill-sar-segmentation --private --source=. --remote=origin --push
git push -u origin develop
git push -u origin app
git push -u origin Quang
git push -u origin Hung
git push -u origin Khoa
```

Neu muon repo public, doi `--private` thanh `--public`.

