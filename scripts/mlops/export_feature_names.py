import json
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath("."))

from src.featurengineering.builder import FeatureBuilder, FeatureConfig
from src.data.loader import load_canonical_data

# Load data
df, df_val, df_test = load_canonical_data()
print(f"Raw data shape: {df.shape}")
print(f"Raw columns ({len(df.columns)}): {list(df.columns)}")

# Build F0
cfg0 = FeatureConfig.from_level(0)
fb0 = FeatureBuilder(cfg0).fit(df)
fs0 = fb0.build_feature_set(df)
f0_cols = list(fs0.contract.feature_names)
print(f"\nF0 feature count: {len(f0_cols)}")

# Build F1
cfg1 = FeatureConfig.from_level(1)
fb1 = FeatureBuilder(cfg1).fit(df)
fs1 = fb1.build_feature_set(df)
f1_cols = list(fs1.contract.feature_names)
print(f"F1 feature count: {len(f1_cols)}")

# Build F2
cfg2 = FeatureConfig.from_level(2)
fb2 = FeatureBuilder(cfg2).fit(df)
fs2 = fb2.build_feature_set(df)
f2_cols = list(fs2.contract.feature_names)
print(f"F2 feature count: {len(f2_cols)}")

# Build F3
cfg3 = FeatureConfig.from_level(3)
fb3 = FeatureBuilder(cfg3).fit(df)
fs3 = fb3.build_feature_set(df)
f3_cols = list(fs3.contract.feature_names)
print(f"F3 feature count: {len(f3_cols)}")

# Compare incremental features
f1_added = [c for c in f1_cols if c not in f0_cols]
f2_added = [c for c in f2_cols if c not in f1_cols]
f3_added = [c for c in f3_cols if c not in f2_cols]

print(f"\nAdded in F1 ({len(f1_added)}): {f1_added}")
print(f"Added in F2 ({len(f2_added)}): {f2_added}")
print(f"Added in F3 ({len(f3_added)}): {f3_added}")

# Detailed analysis of F0
raw_non_date = [c for c in df.columns if c != "Ngày"]
print(f"\nRaw non-date columns count: {len(raw_non_date)}")
print(f"F0 columns count: {len(f0_cols)}")
in_raw_not_f0 = [c for c in raw_non_date if c not in f0_cols]
in_f0_not_raw = [c for c in f0_cols if c not in raw_non_date]
print(f"In raw but not in F0 ({len(in_raw_not_f0)}): {in_raw_not_f0}")
print(f"In F0 but not in raw ({len(in_f0_not_raw)}): {in_f0_not_raw}")

# Save feature_names.json
output_data = {
    "total_features": len(f3_cols),
    "f0_count": len(f0_cols),
    "f1_incremental_count": len(f1_added),
    "f2_incremental_count": len(f2_added),
    "f3_incremental_count": len(f3_added),
    "f0_features": f0_cols,
    "f1_incremental_features": f1_added,
    "f2_incremental_features": f2_added,
    "f3_incremental_features": f3_added,
    "all_70_features": f3_cols
}

out_path = Path("feature_names.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\nSaved feature_names.json to {out_path.resolve()}")
