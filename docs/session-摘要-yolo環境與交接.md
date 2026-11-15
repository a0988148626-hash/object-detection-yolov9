# Session 摘要／交接筆記——物件偵測與影像辨識（YOLO）

給下一個工作階段（例如 Claude Code 或 Colab）接續使用。最後更新：2026-10-31。

## 一、目前狀態總覽

| 項目 | 狀態 |
|---|---|
| 本機 YOLOv9 環境 | 已驗證可用 |
| Mendeley PCB-Defect 資料集 | 已轉換為 YOLO 格式（`pcb_yolo\`），已完成 |
| Mendeley baseline 訓練（30 epochs） | 已完成，`runs\train\pcb_baseline\` |
| Mendeley test 集評估 | **尚未執行** |
| EdgePCB 資料集 | 待下載與確認結構 |
| 焊點／SMT 資料集搜尋與取得 | 待 code 接手 |
| 家禽 YOLO 任務 | 需求未釐清（見第七節） |

## 二、本機環境（已驗證可用）

- 機器：Windows 筆電 `laptop-5k5hfrs0`，**沒有可用的 NVIDIA GPU，為 CPU 訓練**
- Conda 環境：`yolov9`，路徑 `C:\Users\Admin\.conda\envs\yolov9`，Python 3.9.25
- 套件：torch 2.8.0+cpu、opencv 5.0.0，並已依 YOLOv9 倉庫的 `requirements.txt` 補齊（requests、thop、tensorboard、ipython、psutil 等）
- 專案：`C:\Users\Admin\Documents\yolov9_project\yolov9`（原始碼來自 `https://github.com/WongKinYiu/yolov9.git`）
- 權重：`weights\gelan-c.pt`（輕量版，訓練使用）、`weights\yolov9-c-converted.pt`（完整版，推論已驗證）

### 已修正的相容性問題
- `models/experimental.py` 第 243 行，因 PyTorch 2.6+ 預設 `weights_only=True`，載入官方權重會失敗。已改為：
  ```python
  ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)  # load
  ```
- 訓練時 `train.py` 的 `torch.load` 同樣受影響，**不必改程式碼**，改在每個新 shell 設定環境變數即可：
  ```bat
  set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  ```

### 已跑通的驗證
- 推論：`python detect.py --weights weights/yolov9-c-converted.pt --conf 0.25 --img 640 --device cpu --source data/images/horses.jpg`，成功偵測 5 匹馬。
- 訓練 smoke test（1 epoch）與 baseline（30 epochs）皆已完成。

### 已知無害警告
- 畫圖時出現 `FreeTypeFont object has no attribute 'getsize'`，來自 Pillow 10 以後的版本差異，**不影響訓練與指標**。可用 `pip install "pillow<10"` 修正。

## 三、操作規範（避免踩雷）

- **必須用「Anaconda Prompt」**，先 `conda activate yolov9`，出現 `(yolov9)` 才能跑指令。PowerShell 會抓到錯的 Python（沒有 torch）。
- 每個新視窗都要：
  1. `conda activate yolov9`
  2. `cd C:\Users\Admin\Documents\yolov9_project\yolov9`
  3. `set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
- 批次檔（.bat）內容盡量維持純英文，含中文字在部分中文 Windows 上會被 cmd.exe 誤判。
- 資料夾路徑含空格時，一律用雙引號包住。
- PyTorch 目前已推進到 2.9 以後，舊版（如 2.5.1）已從官方索引移除，不建議降版。

## 四、資料集

### 4.1 Mendeley PCB-Defect（已完成）
- 來源：`https://data.mendeley.com/datasets/vdj74sngvn/1`（230 張，6 類，COCO 格式，1704 個框）
- 原始壓縮檔：`C:\Users\Admin\Downloads\PCB-Defect An Annotated Dataset for Surface Defect.zip`（約 154 MB）
- 解壓後標註：`C:\Users\Admin\Downloads\PCB-Defect An Annotated Dataset for Surface Defect\PCB-Defect An Annotated Dataset for Surface Defect\PCB_Defect\PCB_Defect\annotation\_annotations.coco.json`
- 解壓後影像：同層的 `...\PCB_Defect\PCB_Defect\images\`
- 轉換指令（已成功）：
  ```bat
  python pcb_to_yolo.py --ann "<上述 json 路徑>" --img_dir "<上述 images 路徑>" --out pcb_yolo
  ```
- 轉換結果：`yolov9_project\pcb_yolo\`，train 184 / val 23 / test 23，缺檔 0，略過的異常框 0
- 各類框數：mouse_bite 356、spur 296、open_circuit 276、missing_pad 276、short 254、spurious_copper 246
- 授權：頁面顯示 Licence 欄位，**尚未確認內容**，須記錄於 Data Collection SOP

### 4.2 EdgePCB（待處理）
- 來源：Hugging Face `Tanishjain9/industrial-pcb-defect-detection-dataset`（頁面標示 MIT）
- 約 28,000 張，YOLO 格式，640×640
- **資料檔結構未確認**：頁面顯示 Dataset Viewer 找不到資料檔，需下載後實際檢查
- 下載指令（本機）：
  ```bat
  pip install -U huggingface_hub
  huggingface-cli download Tanishjain9/industrial-pcb-defect-detection-dataset --repo-type dataset --local-dir C:\Users\Admin\Documents\yolov9_project\edgepcb
  ```
- 注意：`pcb_to_yolo.py` 的 `--yolo_src` 模式**尚未測試**，目前只支援 `images/labels` 或 `split/images` 兩種結構，需依實際結構調整。
- 訓練量大，本機 CPU 上建議先用子集（例如 2,000 張）做 baseline。

### 4.3 焊點／SMT 相關公開資料集（待 code 接手）
| 資料集 | 內容 | 授權／限制 | 優先度 |
|---|---|---|---|
| Ülger et al. 焊點資料集（IEEE TIM 2023，GitHub `furkanulger/solder-joint-dataset`） | 正常與多種 SMT 焊接錯誤 | 需確認授權 | 高 |
| SolderNet（Moog，arXiv 2211.10274） | 2690 張焊點，含冷焊、燒焦、助焊劑殘留 | **內部資料，公開狀態不明**，需聯絡作者 | 高（類別最接近） |
| SolDef_AI（J. Manuf. Mater. Process.） | 焊接缺陷，Mask R-CNN 實例分割 | 開源，需確認授權 | 中 |
| PCB-AoI（KubeEdge-Ianvs／Kaggle） | 錫膏印刷檢測（SPI），Pascal VOC XML | 需確認授權 | 中（前段製程） |
| DeepPCB | 裸板缺陷，成對模板圖 | **僅限研究用途** | 低 |
| MVTec AD／AD 2 | 通用異常檢測，含刮傷、污染 | **CC BY-NC-SA，禁止商業用途** | 只作方法參考 |

### 4.4 本機其他相關資料（非本次主線）
- `C:\Users\Admin\Downloads\object\object\{cat,Dog,persan}`：貓狗人分類資料，**無 bounding box**，只能分類，先前用於 YOLOv1 推論測試
- `C:\Users\Admin\Downloads\trained_YOLO_model\`：先前 YOLOv1（非 v9）訓練的舊 checkpoint，VOC 20 類，與家禽無關
- 免費資料來源清單：`docs/免費物件偵測影像辨識資料來源.md`

## 五、訓練結果

### Smoke test
- 輸出：`runs\train\pcb_smoke2\`，1 epoch，流程驗證通過（指標低屬正常）

### Baseline（Mendeley，30 epochs，CPU）
- 輸出：`runs\train\pcb_baseline\`（含 `best.pt`、`last.pt`）
- 驗證集（23 張）最後一輪：mAP50 **0.860**、mAP50-95 **0.439**、Precision 0.924、Recall 0.818
- 各類 mAP50：short 0.967、spurious_copper 0.965、spur 0.927、missing_pad 0.898、open_circuit 0.890、**mouse_bite 0.512（Recall 0.432，最弱）**
- **注意**：這是驗證集，驗證集同時用於挑選模型，不能當成最終成績。

### 尚未完成
- **test 集評估**（尚未執行）：
  ```bat
  python val.py --weights runs\train\pcb_baseline\weights\best.pt --data ..\pcb_yolo\pcb.yaml --task test --img 640 --device cpu --name pcb_test
  ```
- 檢查 `runs\train\pcb_baseline\results.csv` 的完整逐 epoch 記錄

## 六、接續訓練與 Colab

### 接續中斷的訓練
```bat
conda activate yolov9
cd C:\Users\Admin\Documents\yolov9_project\yolov9
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python train.py --resume runs\train\pcb_baseline\weights\last.pt
```
`--resume` 會沿用原本的參數與進度，不需要再帶 `--epochs`。中斷時進行中的 epoch 會丟失，只會從最後一個完整 epoch 接續。

### Google Colab（可行，適合 EdgePCB）
- Runtime 選 GPU（T4）
- EdgePCB 在 Colab 內用 `huggingface_hub` 下載；Mendeley 的 zip 上傳到 Google Drive
- `!git clone https://github.com/WongKinYiu/yolov9`，`!pip install -r yolov9/requirements.txt`（Colab 已有 GPU 版 torch，不要裝 CPU 版）
- `gelan-c.pt` 放到 `yolov9/weights/`
- 每個 cell 執行前設定：`os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"`
- 訓練時 `--device 0`
- 結果存到 Google Drive，避免 Colab 閒置斷線遺失
- 免費 GPU 有時數限制，需用 `--resume` 分次訓練
- 報告時須註明執行環境（版本、硬體）

## 七、家禽 YOLO（需求已釐清，見 `家禽YOLO專案規劃與資料來源.md`）

使用者稍早提出「製作家禽的 YOLO」，目標、資料來源已在獨立文件中規劃完成。

相關既有項目：
- 「畜牧資料分析平台」（另一專案，教學／展示性質）中的 `duck_weight_model`，是**體重回歸 CNN**（用影像預測鴨隻體重，PyTorch 移植自 Chen et al. 2023），**不是物件偵測 YOLO**，性質不同。計畫中的第四階段會將兩者串接（YOLO 框出個體 → CNN 估體重）。

## 八、待決定事項（需使用者回覆）
1. EdgePCB baseline 的子集大小（建議 2,000 張），或等待 GPU 環境。
2. 焊點資料集是否納入「冷焊」（外觀判定可靠性有限）。
3. 多視角 R-OR 規則是否採用。
4. Major 缺陷（刮傷、燒焦）是否要求與 Critical 同樣的 Recall 標準。

## 九、合規提醒
- 公開資料集只能用於訓練與方法比較，**不能作為 Validation Report 的正式測試集**。
- 每個外部資料集須記錄來源、版本或 DOI、授權、用途範圍、取得日期與 MD5，寫入 Data Collection SOP。
- 不得宣稱 ISO 13485 規定 mAP 門檻；mAP 目標只能標示為內部驗收標準。
- 是否屬於 SaMD，需先與 TFDA 分類確認。
- 驗證草稿（TC-MV-01～03、TC-DG-01～03，對應 ISO 14971 風險 R-01～R-05）已完成初稿，約需 413 件實體件（不含冷焊）。

## 十、給接手者的注意事項
- 本次的 Claude 電腦控制權限不穩定，Anaconda Prompt 常被隱藏或只能點擊，**不要依賴電腦控制執行指令**，改請使用者在 Anaconda Prompt 執行，或在 Colab 中執行。
- 每次跑指令前確認：`conda activate yolov9`、切到 `yolov9` 目錄、設定環境變數。
- 下載大型資料集前，先確認磁碟空間與使用者同意。
