# =========================
# Config
# =========================
PY      := python
MASS    ?= 200
CONFIG ?= configs/path.yaml

SPLITS_DIR := ml_outputs/splits
MODEL_DIR  := ml_models/sig$(MASS)
WORK_DIR  := ml_models/work
FINAL_DIR  := ml_models/final

# =========================
# Phony targets
# =========================
.PHONY: all help data splits train ablation optimal freeze infer summarize plots ci clean clean-all smoke

# =========================
# Default target
# =========================
all: data splits train ablation optimal freeze infer summarize

smoke: clean
	@echo ">>> CI smoke run (CONFIG=$(CONFIG), MASS=$(MASS))"
	$(PY) -m src.make_dataset --config $(CONFIG)
	$(PY) -m src.prepare_ml --write-splits
	$(PY) -m src.train_bdt --mass $(MASS)
	$(PY) -m src.predict --mass $(MASS) --input $(SPLITS_DIR)/test_sig$(MASS).parquet --split test
	$(PY) -m src.summarize_inference --mass $(MASS)


help:
	@echo "Usage:"
	@echo "  make all            # run full pipeline"
	@echo "  make data           # ROOT -> parquet"
	@echo "  make splits         # train/val/test splits"
	@echo "  make train          # train BDT"
	@echo "  make ablation       # feature ablation (drop1)"
	@echo "  make optimal        # select optimal feature set"
	@echo "  make freeze         # freeze final model"
	@echo "  make infer          # inference on test split"
	@echo "  make summarize      # summarize inference"
	@echo "  make ci             # run pipeline on sampledata"
	@echo ""
	@echo "Variables:"
	@echo "  MASS=200 make all"
	@echo "  CONFIG=configs/path.yaml make data"

# =========================
# Pipeline steps
# =========================

# 1) Dataset creation
data:
	@echo ">>> make_dataset (CONFIG=$(CONFIG))"
	$(PY) -m src/make_dataset --config $(CONFIG)

# 2) Prepare ML splits
splits:
	@echo ">>> prepare_ml (write splits)"
	$(PY) -m src/prepare_ml --write-splits

# 3) Train BDT
train:
	@echo ">>> train_bdt (MASS=$(MASS))"
	$(PY) -m src/train_bdt --mass $(MASS)

# 4) Feature ablation
ablation:
	@echo ">>> feature_ablation (drop1)"
	$(PY) -m src/feature_ablation --mass $(MASS) --mode drop1

# 5) Optimal feature selection
optimal:
	@echo ">>> run_optimal_ablation"
	$(PY) -m src/run_optimal_ablation --mass $(MASS)

# 6) Freeze final model
freeze:
	@echo ">>> freeze_final to latest verion"
	$(PY) -m src/freeze_final --mass $(MASS)

# 7) Inference
infer:
	@echo ">>> predict (test split)"
	$(PY) -m src/predict \
		--mass $(MASS) \
		--input $(SPLITS_DIR)/test_sig$(MASS).parquet \
		--split test

# 8) Summarize inference
summarize:
	@echo ">>> summarize_inference"
	$(PY) -m src/summarize_inference --mass $(MASS)

# =========================
# Convenience targets
# =========================

plots:
	@echo ">>> plot_bdt_diagnostics"
	$(PY) -m src/plot_bdt_diagnostics --mass $(MASS) --modeldir $(MODEL_DIR)

ci:
	@echo ">>> CI quick run on sampledata"
	$(MAKE) smoke CONFIG=configs/path.ci.yaml MASS=200

clean:
	@echo ">>> Cleaning outputs ((keeping final releases))"
	rm -rf ml_outputs root_outputs $(MODEL_DIR) $(WORK_DIR) ml_ablation || true

clean-all:
	@echo ">>> Cleaning outputs ((keeping final releases))"
	rm -rf ml_outputs root_outputs ml_models ml_ablation || true

