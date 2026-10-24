.PHONY: install test demo data v1 v2 stage1 eval coil export docker

install:        ## python deps
	pip install -r requirements.txt

test:           ## unit tests
	python -m pytest -q

demo:           ## run the dashboard locally
	streamlit run app.py

data:           ## NEU-DET -> YOLO layout
	python scripts/build_dataset.py --src NEU-DET --dst YOLO

v1:             ## baseline detector (30 epochs, yolov8n)
	python scripts/train.py --model yolov8n.pt --epochs 30 --name v1_n30 --aug default && cp models/v1_n30.pt models/best_v1.pt

v2:             ## bigger model, longer, strong augmentation
	python scripts/train.py --model yolov8s.pt --epochs 100 --name v2_s100 --aug strong

stage1:         ## anomaly screen bank + calibration
	python scripts/train_anomaly.py --data YOLO --out models/anomaly_bank.pt

eval:           ## metrics, plots, per-class thresholds
	python scripts/evaluate.py --weights models/best_v1.pt

coil:           ## replay val images as a coil into the store
	python scripts/simulate_coil.py --images YOLO/val/images --coil COIL-DEMO-001

export:         ## ONNX + benchmark
	python scripts/export.py --weights models/best_v1.pt --format onnx --bench

docker:
	docker build -t steelsight . && docker run -p 8501:8501 steelsight
