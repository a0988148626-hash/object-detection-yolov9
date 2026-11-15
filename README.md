# 物件偵測與影像辨識專案（YOLOv9）

個人作品集專案：在 Windows + Anaconda 的本機 CPU 環境下，建置 YOLOv9 物件偵測 pipeline，並以 PCB 表面缺陷偵測為第一個完整案例，規劃延伸至家禽（雞/鴨/鵝）偵測與異常監測。

## 專案狀態

| 項目 | 狀態 |
|---|---|
| 本機 YOLOv9（CPU）環境建置 | ✅ 已驗證可用 |
| Mendeley PCB-Defect 資料集轉換為 YOLO 格式 | ✅ 已完成（`pcb_to_yolo.py`） |
| PCB baseline 訓練（30 epochs） | ✅ 已完成，驗證集 mAP50 0.860 |
| PCB test 集最終評估 | ⏳ 待執行 |
| EdgePCB 資料集（Hugging Face） | ⏳ 待下載與整理 |
| 焊點／SMT 缺陷資料集 | ⏳ 搜尋中（見文件） |
| 家禽 YOLO（偵測／分類／異常偵測） | 📋 需求與資料來源已規劃，尚未開始蒐集資料 |

詳細細節見 [`docs/`](docs/) 資料夾。

## 內容物

- `pcb_to_yolo.py` — 將 Mendeley PCB-Defect 資料集（COCO JSON 標註）轉換為 YOLO txt 格式的自製腳本，支援 train/val/test 切分。
- `setup_yolov9_conda.bat` — 在 Windows + Anaconda 下建置 `yolov9` conda 環境的自動化腳本。
- `docs/環境建置指南.md` — YOLOv9 在 Anaconda + VS Code 本機環境建置的完整步驟。
- `docs/session-摘要-yolo環境與交接.md` — 環境細節、已修正的相容性問題、資料集處理進度、訓練結果、待辦事項的完整交接筆記。
- `docs/家禽YOLO專案規劃與資料來源.md` — 家禽偵測/分類/異常監測子專案的目標與可用公開資料來源整理。
- `docs/免費物件偵測影像辨識資料來源.md` — 台灣即時影像來源與常見公開物件偵測資料集整理。

> 本 repo **只包含程式碼與文件**，不包含訓練資料（`pcb_yolo/`）、訓練輸出（`runs/`）、模型權重（`*.pt`），以及第三方原始碼（`yolov9/`，見下方「相依的第三方專案」）——這些檔案較大，且多數可從原始來源重新取得或重新產生。

## 環境需求

- Windows + Anaconda（或 Miniforge，見下方「關於 Anaconda 授權」）
- Python 3.9
- PyTorch（CPU 版即可；有 GPU 可大幅加速訓練）
- 完整步驟見 [`docs/環境建置指南.md`](docs/環境建置指南.md)

## 相依的第三方專案：YOLOv9

本專案使用 [`WongKinYiu/yolov9`](https://github.com/WongKinYiu/yolov9) 作為偵測模型架構，**未將其原始碼納入此 repo**（避免內嵌第三方 git repo、並保持本 repo 精簡）。使用前請自行 clone：

```bash
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```

### 已知相容性修正

PyTorch 2.6 以後預設 `torch.load(..., weights_only=True)`，會導致官方權重載入失敗。需要套用以下修正：

**`models/experimental.py` 第 243 行附近**，將：
```python
ckpt = torch.load(attempt_download(w), map_location='cpu')
```
改為：
```python
ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)
```

**訓練時（`train.py`）** 不需要改程式碼，在每個新的終端機視窗設定環境變數即可：
```bat
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

完整背景與其他已驗證可行的操作細節，見 [`docs/session-摘要-yolo環境與交接.md`](docs/session-摘要-yolo環境與交接.md)。

## 關於 Anaconda 授權

Anaconda Distribution／`defaults` channel 對 200 人以上的營利、非營利或政府組織需要付費商業授權；`conda` CLI 本身與 `conda-forge` channel 對任何規模的使用都仍是免費的。若要完全避開授權風險，建議改用 [Miniforge](https://github.com/conda-forge/miniforge)（預設走 `conda-forge`）。

## 授權

本 repo 中由作者自行撰寫的程式碼（`pcb_to_yolo.py`、`setup_yolov9_conda.bat` 等）採用 MIT License，見 [`LICENSE`](LICENSE)。上述第三方 YOLOv9 原始碼有其自身授權條款，請參考該專案本身。各訓練資料集的授權條款請見 `docs/` 中的說明，使用前請自行確認是否符合您的用途。
