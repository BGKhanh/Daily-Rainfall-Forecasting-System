# Storage Operations Runbook: Two-DB Convention & NASA POWER Vintage Store

**Schema**: DDL V3.2 (`src/data/vintage_store.py`) — SQLite `3.35+` / Turso `libSQL` compatible  
**Canonical Station Lock**: `location_id = "HCMC"` (`TP. Hồ Chí Minh`, `10.7769°N, 106.7009°E`)

---

## 1. Quy ước Hai Cơ sở Dữ liệu (Two-DB Convention)

Hệ thống tách biệt tuyệt đối **DB Demo đóng gói sẵn** (cho CI, Docker smoke-test, và giao diện UI showcase) khỏi **DB Vận hành / Production** (cho bằng chứng forward-test thực tế):

| Thuộc tính | **1. Bundled Demo DB** (`artifacts/models/demo_rainfall_vintage.db`) | **2. Production / Live DB** (`data/vintage/rainfall_vintage.db` hoặc `$VINTAGE_DB_PATH`) |
| :--- | :--- | :--- |
| **Vị trí lưu trữ** | Tracked qua **Git LFS** (`.gitattributes`: `artifacts/models/*.db`, kèm auto-rebuild fallback `scripts/mlops/ensure_demo_db.py`) & đóng gói sẵn vào Docker image `/app/artifacts/models/demo_rainfall_vintage.db` | Bên ngoài container (volume mount hoặc đường dẫn local gitignored) |
| **Chức năng** | Demo API (`/` SPA UI, `/predict`, `/scores`, `/models`, `/health`), parity 2.184 Test origins & CI không cần NASA POWER | Lưu chuỗi quan sát NRT tự thu thập từ ngày go-live (chỉ chứa `observations_vintage`, `forecasts_ledger = 0` cho đến khi phát dự báo forward thực tế) |
| **Nội dung đóng gói** | Bản chụp ngày **2026-11-15** (capture **2026-11-18**): `9,450` dòng quan sát (`2001-01-01..2026-11-15`) + **`4,270` dòng dự báo hồi cứu `POST_TEST`** (`2026-05-01..2026-08-30`) | `9,450` dòng quan sát (`2001-01-01..2026-11-15`), **`0` dòng hồi cứu trong `forecasts_ledger`**; chỉ chạy lệnh `ingest_nasa_vintage.py` để tích lũy quan sát mới |
| **Chế độ mở & Journal** | Nhân bản từ Production DB qua `VACUUM INTO`, chạy `run_forecast_ledger.py` trên bản Demo rồi chốt `finalize_db_for_read_only()` (`journal_mode = DELETE`); mở `?mode=ro` | Ghi qua `WAL` → cuối mỗi giao dịch tự động chạy `finalize_db_for_read_only()` (`wal_checkpoint(TRUNCATE)` + `journal_mode = DELETE`) |
| **Giá trị chứng cứ** | Dữ liệu **hồi cứu** (`is_prospective = 0`) — **chỉ DB Demo mới chứa 4.270 dòng hồi cứu** | Chỉ các dự báo tương lai trên Production DB với `julianday(obs_first_ingested_at_utc) > julianday(issued_at_utc)` (`is_prospective = 1`) mới được tính là **forward prospective evidence** |

Thứ tự phân giải tự động trong `src/data/provider.py::resolve_sqlite_db_path_and_role()`:
1. Tham số tường minh `db_path` (nếu truyền) $\rightarrow$ `db_role = "local_or_production_db"`
2. Biến môi trường `VINTAGE_DB_PATH` (nếu cấu hình) $\rightarrow$ `db_role = "local_or_production_db"`
3. `data/vintage/rainfall_vintage.db` (nếu tồn tại trên ổ đĩa) $\rightarrow$ `db_role = "local_or_production_db"`
4. `artifacts/models/demo_rainfall_vintage.db` (nếu tồn tại, mặc định trong Docker/CI) $\rightarrow$ `db_role = "bundled_demo_db"`
5. `artifacts/models/sample_recent_history_35d.csv` (chế độ tĩnh khi `WEATHER_DB_BACKEND=sample_csv` hoặc thiếu cả 2 DB) $\rightarrow$ `db_role = "sample_csv"`

---

## 2. Quy trình Khởi tạo & Vận hành Hai DB Đúng Thứ Tự (Round 8)

```bash
# 1. Dựng DB Production sạch (CHỈ chứa quan sát, forecasts_ledger = 0):
#    Historical CSV Seed (9,251 dòng 2001-01-01..2026-04-30) + Quét NASA POWER 210 ngày
python scripts/data/ingest_nasa_vintage.py \
  --db-path data/vintage/rainfall_vintage.db \
  --seed-csv-if-empty \
  --lookback-days 210 \
  --report-json reports/storage_audit/nasa_first_live_ingest.json

# 2. Tạo DB Demo từ Production DB qua VACUUM INTO (lấy phần quan sát sạch)
python -c "from src.data.vintage_store import connect_sqlite, finalize_db_for_read_only; c = connect_sqlite('data/vintage/rainfall_vintage.db', read_only=False); c.execute('VACUUM INTO ?', ('artifacts/models/demo_rainfall_vintage.db',)); finalize_db_for_read_only(c); c.close()"

# 3. Chèn 4.270 dòng hồi cứu POST_TEST vào RIÊNG DB Demo + Chạy kiểm định Parity & Độ nhạy Block (7/14/21)
python scripts/mlops/run_forecast_ledger.py \
  --db-path artifacts/models/demo_rainfall_vintage.db \
  --verify-test-parity \
  --issue-post-test \
  --report-json reports/storage_audit/post_test_retrospective_scores.json
```

---

## 3. Quy tắc Đóng băng Dữ liệu & Kiểm soát WAL

1. **Không bao giờ dùng `?immutable=1`**: Thực nghiệm DDL V3.2 chứng minh nếu `WAL` chưa checkpoint, `?mode=ro&immutable=1` hoặc sao chép riêng file `.db` sẽ bỏ qua `.db-wal` và đọc thiếu dòng vừa commit. Mọi hàm ghi đều kết thúc bằng `finalize_db_for_read_only(conn)` và `connect_sqlite(read_only=True)` chỉ mở bằng `file:{path}?mode=ro`.
2. **Làm sạch để đóng gói Demo DB**:
   ```bash
   python -c "from src.data.vintage_store import connect_sqlite, finalize_db_for_read_only; c = connect_sqlite('data/vintage/rainfall_vintage.db', read_only=False); c.execute(\"VACUUM INTO 'artifacts/models/demo_rainfall_vintage.db'\"); c.close()"
   ```
   Sau đó cập nhật digest SHA-256 trong `artifacts/models/metadata.json["sha256_checksums"]["demo_rainfall_vintage.db"]`.
3. **Bất biến ngày neo (`COMPLETE_ALL_35`) & Độ trễ Bức xạ**:
   - **8 biến CERES SYN1deg** trễ ~**78 ngày** (`last_valid_date = 2026-08-30` ở lần quét 2026-11-16/17).
   - **3 biến FLASHFlux** có độ trễ 4–5 ngày và từng ghi nhận **1 ngày khuyết nội bộ (interior gap)** tại **`2026-11-06`** (nhãn `FLASHFLUX_INCOMPLETE`), khiến cả cửa sổ vẫn hiển thị `per_variable_last_valid_date` nhưng `incomplete_dates_in_window` đánh dấu rõ ngày thiếu.
   - Các ngày NASA POWER chưa công bố (`missing_count == 35`) bị bỏ qua (`rows_unpublished_skipped`), không ghi rác vào `observations_vintage`.
   - `POST /predict` từ chối mọi `anchor_date` không đạt `COMPLETE_ALL_35` (HTTP `422 [INCOMPLETE_RADIATION_AT_ANCHOR]`).
4. **Chuẩn hóa UTC ISO-8601 & So sánh `julianday()`**: Mọi timestamp ghi vào DB được chuẩn hóa về `YYYY-MM-DDTHH:MM:SS.ffffffZ` qua `normalize_utc_iso()`, và `v_forecast_scores` + `query_observations_as_of()` so sánh qua `julianday(...)` để miễn nhiễm với mọi biến thể chuỗi ISO (`' '`, `'T'`, `'Z'`, `'+07:00'`).
5. **Mốc Chuyển Nguồn Thực Đo (`audit_probe` trong `ingestion_runs`)**: Thí nghiệm đơn ngày NASA POWER Daily API (`v2.10.4`, chạy `2026-11-19`) xác nhận `header.sources` đổi từ `MERRA2 → GEOSIT` sau **`2026-09-30`** (trễ **50 ngày**) và `SYN1DEG → FLASHFLUX` sau **`2026-08-30`** (trễ **81 ngày**). Script `scripts/audit/probe_nasa_header_sources.py` chèn một dòng `run_type = 'audit_probe'` (bảo toàn khóa 3 nhóm latency + bổ sung `"source_transition_cutoffs"`) để tuân thủ trigger `trg_ingestion_runs_no_update` và phục vụ phân loại `FINALIZED` vs `PROVISIONAL` cho `GET /observations`.
6. **Khôi phục DB Demo từ Git LFS Fallback**: Trong môi trường CI hoặc khi clone thiếu `git lfs pull`, chạy `python scripts/mlops/ensure_demo_db.py` để phát hiện LFS pointer và tự động tái tạo `demo_rainfall_vintage.db` hợp lệ trước khi chạy test hoặc `docker build`.

---

## 4. Schema Migration `V3.3`, Chế độ Turso (`STORE_BACKEND`) & Pipeline Tự động (Round 11)

1. **DDL `V3.3` & Provenance `config_sha256`**:
   - Bảng `schema_migrations` (`V3.3`) xác nhận đã thêm cột `config_sha256` cho `ingestion_runs` và `forecasts_ledger` cùng composite index `idx_obs_vintage_loc_date_ingested` trên `(location_id, obs_date, ingested_at_utc, version_num DESC)`.
   - Các truy vấn API `/observations`, `/health`, `/locations`, `/predict`, `/scores` đẩy bộ lọc ngày trực tiếp xuống bảng gốc thay vì quét toàn bộ view `v_observations_latest`, kết hợp cache trong bộ nhớ theo khóa `(db_identifier, latest_run_id, max_forecast_rowid)`.
2. **Ba chế độ `STORE_BACKEND` (`src/data/turso_backend.py`)**:
   - `local` (mặc định trong CI & Docker, không cần token): đọc SQLite local (`rainfall_vintage.db` / `demo_rainfall_vintage.db`).
   - `turso_replica`: đồng bộ incremental từ Turso về file replica cục bộ (`TURSO_REPLICA_PATH`, mặc định `data/vintage/turso_replica.db`) khi khởi động và theo chu kỳ nền `300s ± 30s` jitter; API đọc từ replica cục bộ với `0` `rows_read` từ xa mỗi request người dùng.
   - `turso_remote`: truy vấn trực tiếp qua Turso Hrana HTTP v2 (`/v2/pipeline`, ưu tiên `TURSO_READONLY_AUTH_TOKEN` cho kết nối read-only) với batch transaction nguyên tử (`BEGIN IMMEDIATE ... COMMIT / ROLLBACK`).
   - **Quy tắc Tách hai DB với Turso**: Production DB (`rainfall_vintage.db` — **chỉ chứa quan sát, `forecasts_ledger = 0`**) được đồng bộ lên Turso; Demo DB (`artifacts/models/demo_rainfall_vintage.db` chứa `4,270` dòng hồi cứu) luôn giữ **strictly local**. Chi tiết định lượng xem [`docs/TURSO_COST_AND_OPERATIONS.md`](TURSO_COST_AND_OPERATIONS.md).
3. **CLI Pipeline Tự động Idempotent (`scripts/mlops/automated_pipeline.py`) & GitHub Actions Cron (`.github/workflows/scheduled_ingest.yml`)**:
   - `probe-headers`: dò ranh giới `header.sources` theo từng ngày và ghi nhận bản ghi `audit_probe` mới (kèm `config_sha256`).
   - `ingest-window`: quét cửa sổ NASA POWER (mặc định 210 ngày) bằng giao dịch nguyên tử; ngày trùng hash không tạo bản ghi mới.
   - `issue-forecasts`: chỉ phát hành dự báo cho ngày neo `COMPLETE_ALL_35` chưa đủ 35 dòng (5 mô hình × 7 ngày); tự động trả trạng thái `SKIPPED` (`no_new_complete_anchor`) khi chưa có ngày neo mới và **không bao giờ chạy `UPDATE` trên `forecasts_ledger`**.
   - `refresh-cache`: đồng bộ replica (nếu bật), xóa cache cũ, làm ấm `state_key` và gọi webhook làm mới tùy chọn (`CACHE_REFRESH_WEBHOOK_URL`).
   - Workflow `.github/workflows/scheduled_ingest.yml` chạy vào `20:23 UTC` hằng ngày dưới nhóm khóa `concurrency: ds108-vintage-ingest` (`cancel-in-progress: false`). Lưu ý: theo tài liệu chính thức GitHub Actions, lịch cron hoạt động theo cơ chế best-effort (có thể trễ hoặc bị bỏ qua khi tải cao) và tự động tắt sau 60 ngày repo công khai không có commit mới.
