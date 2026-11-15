# 家禽 YOLO 專案規劃與交接筆記

給接手的 Claude Code 使用。這是新專案的需求釐清 + 可用資料來源整理，銜接前一份 [session 摘要——YOLO 環境與交接](session-摘要-yolo環境與交接.md) 裡已經跑通的本機 YOLOv9 環境（`C:\Users\Admin\Documents\yolov9_project\yolov9`，conda 環境 `yolov9`）。

## 專案目標（使用者已確認，共四項，可能分階段實作）

1. **物件偵測**——框出家禽位置、計數
2. **疾病／異常狀態偵測**——例如跛腳、羽毛異常等
3. **品種／物種分類**——區分雞、鴨、鵝等不同家禽
4. **跟「畜牧資料分析平台」的白鴨體重模型整合**——用 YOLO 先框出鴨隻個體，再接現有的 `duck_weight_model`（體重回歸 CNN，PyTorch，訓練自 Chen et al. 2023 論文的 50 隻白菜鴨 × 80 張照片資料集）做體重估算

## 資料現況

目前**沒有任何訓練資料**（不管標註好的還是原始影像都沒有），需要從頭尋找/建立。以下是初步找到、可作為起點的公開資源：

### 物件偵測（框位置/計數）——雞

| 資料集 | 說明 | 連結 |
|---|---|---|
| Chicken Detection and Tracking | 雞隻偵測+追蹤，已標註 bounding box | [Roboflow Universe](https://universe.roboflow.com/chickens/chicken-detection-and-tracking) |
| chicken count | 雞隻計數用資料集 | [Roboflow Universe](https://universe.roboflow.com/chickendetection-sct5j/chicken-count/dataset/4) |
| Poultry-Chicken | 家禽/雞隻偵測 | [Roboflow Universe](https://universe.roboflow.com/livestock/poultry-chicken) |
| chicken-yolo | 已整理成 YOLO 格式 | [Roboflow Universe](https://universe.roboflow.com/od-3hysf/chicken-yolo) |
| CHICKEN[YOLO] | 實例分割（比 bounding box 更精細） | [Roboflow Universe](https://universe.roboflow.com/projects-mmpxg/chicken-yolo-tpynr) |
| Roboflow "poultry" / "chicken" 全站搜尋 | 還有更多可篩選 | [class:poultry](https://universe.roboflow.com/search?q=class%3Apoultry)、[class:chicken](https://universe.roboflow.com/search?q=class%3Achicken) |

### 物件偵測——鴨

目前沒找到現成、大規模、公開的「鴨隻 bounding box 標註資料集」；比較實際的路線：
- 參考這篇教學自己標註／訓練：[Building a Duck Detector with YOLO（Evan Tan, Medium）](https://medium.com/@evanjtan/building-a-duck-detector-with-yolo-9b265af4a83d)
- 直接沿用「畜牧資料分析平台」裡已經有的白鴨影像（50 隻 × 80 張，來自 Chen et al. 2023 論文/GitHub `RuoyuChen10/Image_weighing`）——這批影像目前應該只有體重標籤、沒有 bounding box，需要另外用 LabelImg 或 Roboflow 手動標註（不過因為鴨隻通常佔滿畫面/單隻入鏡，如果目的只是「確認有沒有鴨」而非精準定位，也可以考慮先跳過偵測直接用分類模型）

### 疾病／異常狀態偵測

| 資料集 | 說明 | 連結 |
|---|---|---|
| Poultry Pathology Visual Dataset | 家禽病理影像 | [Kaggle](https://www.kaggle.com/datasets/chandrashekarnatesh/poultry-diseases) |
| Chicken Disease Image Classification | 雞糞影像分類判斷疾病（分類，非偵測） | [Kaggle](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)、[GitHub](https://github.com/David-Ademola/Chicken-Disease-Image-Classification) |
| chicken-health-behavior-multimodal | 雞隻健康行為多模態資料集 | [Hugging Face](https://huggingface.co/datasets/IceKhoffi/chicken-health-behavior-multimodal) |
| Roboflow "sick chicken" 搜尋 | 病雞相關資料集 | [class:sick chicken](https://universe.roboflow.com/search?q=class%3Asick+chicken) |
| LameDetect（論文） | YOLO+CNN 混合模型做跛腳早期偵測，可參考方法論 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169925013456) |

**注意**：目前找到的疾病相關資料多半是「分類」（一張圖判斷健康/生病），不是「偵測」（框出哪隻異常），如果要做到「偵測」層級，可能需要在既有分類資料集上自己補標註，或先用分類模型當作第一階段。

## 建議的分階段做法

1. **第一階段**：用 Roboflow 上的雞隻偵測資料集（YOLO 格式現成可下載），直接用已經跑通的本機 YOLOv9 環境訓練一個「雞隻偵測 + 計數」模型，快速拿到一個能動的 baseline
2. **第二階段**：擴充多類別（雞/鴨/鵝品種分類）——需要額外蒐集鴨、鵝的標註資料，或用第一階段模型 + 少量鴨鵝資料做遷移學習
3. **第三階段**：疾病/異常偵測——先評估用分類模型（判斷單張圖是否異常）還是偵測模型（框出異常個體），取決於實際應用情境（例如是空拍/監視器廣角畫面 vs. 近拍單隻）
4. **第四階段**：跟 `duck_weight_model` 整合——用偵測模型框出鴨隻 → 裁切（crop）→ 餵進現有的體重回歸 CNN，形成「偵測 + 體重估算」的完整管線

## 開放問題（需要使用者進一步確認才能往下規劃）

- 家禽的實際應用場景是什麼？（畜牧場監視器空拍/近拍？手機拍照？影片還是靜態照片？）
- 品種範圍：只要雞鴨鵝，還是要更細的品種（例如白菜鴨 vs. 其他鴨種）？
- 疾病偵測要偵測到什麼精細度：整批健康度評估，還是要點出個別異常的鴨/雞？
- 跟白鴨體重模型整合時，資料集是否就直接沿用現有的 50 隻 × 80 張，還是需要重新蒐集？
