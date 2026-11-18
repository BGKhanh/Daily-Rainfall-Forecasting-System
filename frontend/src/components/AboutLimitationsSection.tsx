import React from "react";
import { APP_CONFIG } from "../api/client";
import type { HealthResponse } from "../types/api";

interface Props {
  health: HealthResponse | null;
}

export const AboutLimitationsSection: React.FC<Props> = ({ health }) => {
  return (
    <section id="gioi-thieu-va-gioi-han" className="panel" aria-labelledby="heading-about">
      <div className="panel-header">
        <div>
          <h2 id="heading-about">4. Giới thiệu mô hình, quản trị và giới hạn hệ thống</h2>
          <p className="panel-subtitle">
            Minh bạch về phạm vi khoa học, lựa chọn mô hình, nguồn dữ liệu và giới hạn triển khai.
          </p>
        </div>
      </div>

      <div className="limitations-grid">
        <div className="info-card">
          <h3>Kiến trúc & Đóng băng tham số</h3>
          <ul>
            <li>
              <strong>Mô hình hồi quy điểm trực tiếp (Direct Multi-Horizon Regression)</strong> cho
              lượng mưa tích lũy theo ngày (mm/ngày) trên 7 bước <code>h=1..7</code> tại TP. Hồ Chí
              Minh (10.7769°N, 106.7009°E).
            </li>
            <li>
              <strong>Đóng băng tham số từ 04/2018</strong>: Bộ chuyển đổi đặc trưng{" "}
              <code>FeatureBuilder</code> và tham số mô hình được huấn luyện cố định trên tập{" "}
              <code>Dev Train</code> (<code>2001-01-01</code> đến <code>2018-04-30</code>).
            </li>
            <li>
              <strong>Không suy diễn lặng lẽ (Zero Silent Imputation)</strong>: Yêu cầu đủ 35 biến
              thực đo/vệ tinh liên tục; mọi chuỗi khuyết dữ liệu hoặc sai trạm đều trả lỗi rõ ràng.
            </li>
          </ul>
        </div>

        <div className="info-card">
          <h3>Quản trị lựa chọn mô hình & Phơi bày Test</h3>
          <ul>
            <li>
              <strong>Champion chọn theo Validation</strong>: <strong>SC-30</strong>{" "}
              (<code>LightGBM Tweedie F2 Config A</code>, alias <code>@shadow_val_champ</code>).
            </li>
            <li>
              <strong>Mô hình phục vụ chính thức (<code>@serving</code>)</strong>: <strong>SC-35</strong>{" "}
              (<code>XGBoost Tweedie F3</code>) —{" "}
              <strong>do chủ dự án chọn sau khi đã xem kết quả tập Test</strong>.
            </li>
            <li>
              <strong>Shadow Contender</strong>: <strong>SC-31</strong>{" "}
              (<code>LightGBM Tweedie F3 Config A</code>, alias <code>@shadow_test_contender</code>).
            </li>
          </ul>
        </div>

        <div className="info-card">
          <h3>Phạm vi dữ liệu & Liên kết cấu hình</h3>
          <ul>
            <li>
              Mọi điểm số và chuỗi hồi cứu đều đánh giá trên{" "}
              <strong>dữ liệu đã hoàn chỉnh (<code>COMPLETE_ALL_35</code>)</strong>.
            </li>
            <li>
              Cơ sở dữ liệu hiển thị trong container hiện là{" "}
              <strong>bản chụp demo (<code>{health?.db_role ?? "bundled_demo_db"}</code>)</strong> cập nhật
              quan sát tới ngày <code>{health?.data_snapshot_date ?? "2026-11-15"}</code>.
            </li>
            <li>
              <strong>Không bao gồm</strong>: mô phỏng giả định (what-if), đa địa điểm, bản đồ tương
              tác, xác suất mưa phân loại, cảnh báo thiên tai, hoặc nạp dữ liệu live từ trình duyệt.
            </li>
          </ul>
          <div className="links-row">
            <a href={APP_CONFIG.repoUrl} target="_blank" rel="noreferrer">
              Kho mã nguồn GitHub
            </a>
            <a href={APP_CONFIG.ciUrl} target="_blank" rel="noreferrer">
              GitHub Actions CI
            </a>
            <a href={APP_CONFIG.mlflowUrl} target="_blank" rel="noreferrer">
              DagsHub MLflow Tracking
            </a>
          </div>
        </div>
      </div>
    </section>
  );
};
