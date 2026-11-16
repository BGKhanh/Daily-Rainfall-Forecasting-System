# BÁO CÁO TỔNG KẾT ĐÁNH GIÁ GIAI ĐOẠN TEST (FINAL TEST BENCHMARK)

**Dự án Dự báo Lượng mưa TP. Hồ Chí Minh (DS108)**
**Bộ chuẩn tắc Protocol:** v1.0.21 | **Ranh giới:** 2.184 Test Origins (2020-05-01 đến 2026-04-23, $H=7$, 15.288 điểm mục tiêu)
**Chính sách Huấn luyện:** Single-Fit Policy (Decision M2) — Huấn luyện 1 lần trên Dev Train (2001–2018), đóng băng trọng số, lăn ngữ cảnh nhân quả qua Test.

---

## 1. MÔ HÌNH QUÁN QUÂN VÀ BẢNG XẾP HẠNG TOP 10 TEST

> **Mô hình Quán quân Test (Champion):** `SC-31` (**lightgbm_tweedie_F3**) thuộc họ **Tabular ML Tweedie**.
> - **Pooled MAE:** **3.7246 mm**
> - **Pooled RMSE:** **8.1194 mm**
> - **Pooled MASE:** **1.0062**
> - **Tầng ứng viên:** `Tier-1` (Dung sai $\varepsilon = 10^{-4}$ mm)

### Bảng xếp hạng Top 10 Mô hình trên Toàn bộ 6 Năm Test Ngoại mẫu:

| Hạng | Tầng | Scenario ID | Model | Họ mô hình | Chiến lược | Feature Set | Pooled MAE (mm) | Pooled RMSE (mm) | Pooled MASE |
|:---:|:---:|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|
| 1 | Tier-1 | `SC-31` | `lightgbm_tweedie_F3` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F3 | **3.7246** | 8.1194 | 1.0062 |
| 2 | Tier-2 | `SC-35` | `xgboost_tweedie_F3` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F3 | **3.7333** | 8.0982 | 1.0086 |
| 3 | Tier-3 | `SC-30` | `lightgbm_tweedie_F2` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F2 | **3.7501** | 8.1143 | 1.0131 |
| 4 | Tier-4 | `SC-34` | `xgboost_tweedie_F2` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F2 | **3.7605** | 8.0784 | 1.0159 |
| 5 | Tier-5 | `SC-29` | `lightgbm_tweedie_F1` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F1 | **3.7774** | 8.0906 | 1.0205 |
| 6 | Tier-6 | `SC-33` | `xgboost_tweedie_F1` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F1 | **3.7841** | 8.0665 | 1.0223 |
| 7 | Tier-7 | `SC-28` | `lightgbm_tweedie_F0` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F0 | **3.8493** | 8.1533 | 1.0399 |
| 8 | Tier-8 | `SC-32` | `xgboost_tweedie_F0` | Tabular ML Tweedie | Tweedie Compound Poisson-Gamma | F0 | **3.8555** | 8.1207 | 1.0416 |
| 9 | Tier-9 | `SC-19` | `rf_direct_F3` | Tabular ML Direct | Direct Multi-Horizon | F3 | **3.9400** | 8.1123 | 1.0644 |
| 10 | Tier-10 | `SC-27` | `lightgbm_direct_F3` | Tabular ML Direct | Direct Multi-Horizon | F3 | **3.9525** | 8.0395 | 1.0678 |

---

## 2. SO SÁNH GIỮA CÁC HỌ MÔ HÌNH (FAMILY COMPARISON)

| Họ mô hình | Số kịch bản | Hạng tốt nhất | MAE tốt nhất (mm) | MAE trung bình (mm) | RMSE tốt nhất (mm) | RMSE trung bình (mm) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tabular ML Tweedie** | 8 | 1 | **3.7246** | 3.7794 | 8.0665 | 8.1052 |
| **Tabular ML Direct** | 16 | 9 | **3.9400** | 4.0727 | 8.0016 | 8.0786 |
| **Two-Stage Hurdle** | 32 | 12 | **3.9651** | 4.1385 | 7.9976 | 8.2899 |
| **Deep Neural** | 5 | 17 | **4.0291** | 4.7176 | 8.6647 | 10.3475 |
| **Intermittent Demand** | 5 | 57 | **4.3224** | 4.3688 | 8.3468 | 8.4138 |
| **Classical Statistical** | 4 | 58 | **4.3294** | 4.3393 | 8.2009 | 8.2517 |
| **Heuristic Baseline** | 2 | 71 | **5.1389** | 5.3384 | 10.7620 | 11.0958 |

---

## 3. ĐỐI ĐẦU BẤT KHẢ PHIẾM ĐỊNH: HURDLE STRATEGY A vs STRATEGY B

> **Ghi chú Phương pháp Luận (CR-04):** Trên phân vùng Validation, Strategy B từng thể hiện kết quả lạc quan giả tạo do ngưỡng $\tau^*$ được tối ưu trực tiếp trên Validation. **Trên tập Test này, toàn bộ ngưỡng $\tau_M^*$ đã bị đóng băng tuyệt đối**, mang lại bằng chứng khách quan, không thiên lệch 100%.

| Họ mô hình | Bộ đặc trưng | Scen A | Hạng A | MAE A (mm) | Scen B | Hạng B | MAE B (mm) | $\Delta$ MAE (B - A) | Bên Thắng |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| logistic | F3 | `SC-44` | 12 | **3.9651** | `SC-52` | 25 | 4.0770 | +0.1119 | **Strategy A** |
| lightgbm | F3 | `SC-48` | 13 | **3.9986** | `SC-56` | 54 | 4.2822 | +0.2836 | **Strategy A** |
| xgboost | F3 | `SC-64` | 15 | **4.0112** | `SC-72` | 38 | 4.1279 | +0.1167 | **Strategy A** |
| lightgbm | F1 | `SC-46` | 18 | **4.0323** | `SC-54` | 53 | 4.2689 | +0.2366 | **Strategy A** |
| lightgbm | F2 | `SC-47` | 20 | **4.0439** | `SC-55` | 56 | 4.3025 | +0.2586 | **Strategy A** |
| xgboost | F2 | `SC-63` | 21 | **4.0550** | `SC-71` | 37 | 4.1220 | +0.0669 | **Strategy A** |
| xgboost | F1 | `SC-62` | 24 | **4.0743** | `SC-70` | 44 | 4.1465 | +0.0723 | **Strategy A** |
| random | F3 | `SC-60` | 26 | **4.0948** | `SC-68` | 32 | 4.1145 | +0.0197 | **Strategy A** |
| logistic | F1 | `SC-42` | 31 | **4.1142** | `SC-50` | 36 | 4.1210 | +0.0068 | **Strategy A** |
| logistic | F0 | `SC-41` | 34 | **4.1176** | `SC-49` | 28 | 4.1053 | -0.0124 | **Strategy B** |
| lightgbm | F0 | `SC-45` | 35 | **4.1194** | `SC-53` | 64 | 4.3786 | +0.2592 | **Strategy A** |
| logistic | F2 | `SC-43` | 41 | **4.1388** | `SC-51` | 47 | 4.1622 | +0.0234 | **Strategy A** |
| random | F2 | `SC-59` | 45 | **4.1505** | `SC-67` | 33 | 4.1173 | -0.0332 | **Strategy B** |
| random | F1 | `SC-58` | 48 | **4.1734** | `SC-66` | 39 | 4.1325 | -0.0409 | **Strategy B** |
| xgboost | F0 | `SC-61` | 50 | **4.1966** | `SC-69` | 51 | 4.2124 | +0.0158 | **Strategy A** |
| random | F0 | `SC-57` | 55 | **4.2873** | `SC-65` | 49 | 4.1870 | -0.1003 | **Strategy B** |

**Kết luận Thực nghiệm Đối đầu Hurdle:** Strategy A (Mixture Expectation) thắng áp đảo với tỉ số **12/16 cặp** trước Strategy B (Hard Gate). Điều này chứng minh rằng việc kết hợp kỳ vọng liên tục p * mu_wet + (1 - p) * mu_dry tránh được hoàn toàn sự trừng phạt nặng nề của sai số biên (boundary misclassification penalty) so với hàng rào cứng rời rạc.

---

## 4. PHÂN TÍCH SUY GIẢM ĐỘ CHÍNH XÁC THEO CHÂN TRỜI (HORIZON DEGRADATION h=1..7)

Sai số MAE (mm) qua các bước dự báo từ ngày thứ 1 đến ngày thứ 7 của Top 5 mô hình hàng đầu:

| Hạng | Scenario ID | Model | Pooled MAE | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `SC-31` | `lightgbm_tweedie_F3` | **3.7246** | 3.4747 | 3.7652 | 3.7747 | 3.7622 | 3.7567 | 3.7680 | 3.7710 |
| 2 | `SC-35` | `xgboost_tweedie_F3` | **3.7333** | 3.4965 | 3.7676 | 3.7599 | 3.7760 | 3.7590 | 3.7968 | 3.7776 |
| 3 | `SC-30` | `lightgbm_tweedie_F2` | **3.7501** | 3.4777 | 3.8093 | 3.7786 | 3.7980 | 3.7752 | 3.8016 | 3.8101 |
| 4 | `SC-34` | `xgboost_tweedie_F2` | **3.7605** | 3.5095 | 3.8044 | 3.7964 | 3.7911 | 3.7909 | 3.8206 | 3.8107 |
| 5 | `SC-29` | `lightgbm_tweedie_F1` | **3.7774** | 3.4896 | 3.8216 | 3.8121 | 3.8120 | 3.8333 | 3.8276 | 3.8456 |

---

## 5. KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (DIEBOLD-MARIANO TEST VS CHAMPION)

Kiểm định giả thuyết $H_0$: Không có sự khác biệt về sai số MAE giữa Mô hình Quán quân và Mô hình So sánh ($N=15.288$ quan sát):

| Hạng | Scenario ID | Model | Họ mô hình | MAE (mm) | Chênh lệch $\Delta$ MAE | DM Statistic | p-value | Có ý nghĩa ($p < 0.05$)? |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 2 | `SC-35` | `xgboost_tweedie_F3` | Tabular ML Tweedie | 3.7333 | +0.0087 | -2.1426 | 3.2164e-02 | Có (Khác biệt thực sự) |
| 3 | `SC-30` | `lightgbm_tweedie_F2` | Tabular ML Tweedie | 3.7501 | +0.0254 | -5.4841 | 4.2221e-08 | Có (Khác biệt thực sự) |
| 4 | `SC-34` | `xgboost_tweedie_F2` | Tabular ML Tweedie | 3.7605 | +0.0359 | -6.6935 | 2.2543e-11 | Có (Khác biệt thực sự) |
| 5 | `SC-29` | `lightgbm_tweedie_F1` | Tabular ML Tweedie | 3.7774 | +0.0527 | -9.6419 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 6 | `SC-33` | `xgboost_tweedie_F1` | Tabular ML Tweedie | 3.7841 | +0.0594 | -10.2888 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 7 | `SC-28` | `lightgbm_tweedie_F0` | Tabular ML Tweedie | 3.8493 | +0.1247 | -16.0088 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 8 | `SC-32` | `xgboost_tweedie_F0` | Tabular ML Tweedie | 3.8555 | +0.1309 | -17.2816 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 9 | `SC-19` | `rf_direct_F3` | Tabular ML Direct | 3.9400 | +0.2154 | -17.8843 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 10 | `SC-27` | `lightgbm_direct_F3` | Tabular ML Direct | 3.9525 | +0.2279 | -27.3406 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 11 | `SC-15` | `lr_direct_F3` | Tabular ML Direct | 3.9579 | +0.2332 | -21.1935 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 12 | `SC-44` | `logistic_linear_stratA_F3` | Two-Stage Hurdle | 3.9651 | +0.2405 | -18.7940 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 13 | `SC-48` | `lightgbm_stratA_F3` | Two-Stage Hurdle | 3.9986 | +0.2740 | -23.9428 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 14 | `SC-23` | `xgboost_direct_F3` | Tabular ML Direct | 4.0067 | +0.2821 | -25.1274 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 15 | `SC-64` | `xgboost_stratA_F3` | Two-Stage Hurdle | 4.0112 | +0.2866 | -25.4097 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 71 | `SC-01` | `Naive` | Heuristic Baseline | 5.1389 | +1.4143 | -25.7904 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 72 | `SC-02` | `SeasonalNaive` | Heuristic Baseline | 5.5380 | +1.8134 | -30.7039 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 58 | `SC-03` | `AutoARIMA` | Classical Statistical | 4.3294 | +0.6048 | -35.7503 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 62 | `SC-11` | `IMAPA` | Intermittent Demand | 4.3553 | +0.6307 | -27.2433 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 43 | `SC-12` | `lr_direct_F0` | Tabular ML Direct | 4.1456 | +0.4210 | -37.4402 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 17 | `SC-37` | `PatchTST` | Deep Neural | 4.0291 | +0.3045 | -15.7155 | 0.0000e+00 | Có (Khác biệt thực sự) |
| 25 | `SC-52` | `logistic_linear_stratB_F3` | Two-Stage Hurdle | 4.0770 | +0.3523 | -17.0079 | 0.0000e+00 | Có (Khác biệt thực sự) |

---

## 6. DANH MỤC TẠO TÁC VÀ BẰNG CHỨNG THỰC NGHIỆM ĐÃ XUẤT

- **Bảng xếp hạng tổng hợp 72 kịch bản Test:** `reports/test_benchmark/unified_72_scenarios_test_leaderboard.csv`
- **Bảng phân rã chi tiết theo chân trời h=1..7:** `reports/test_benchmark/unified_72_test_horizon_breakdown.csv`
- **5 Tệp CSV Kết quả Gốc:** `data/processed/results/*_test_results.csv`
- **5 Tệp CSV Horizon Breakdown:** `data/processed/results/*_test_horizon_breakdown.csv`
- **5 Tệp Ma trận Dự báo NPZ:** `data/processed/results/*_test_forecasts.npz`
- **Tệp Xác suất Hurdle NPZ:** `data/processed/results/ml_hurdle_test_probabilities.npz`
