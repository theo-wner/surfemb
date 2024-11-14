# Wissenswertes zum Repo "SurfEmb" für mich als Reminder

## 1. Pytorch-Lightning-Version

Es muss eine ältere Pytorch-Lightning-Version (z.B. 1.4.0) installiert werden
z.B.: pip install pytorch-lightning==1.4.0

## 2. Ordnerstruktur (für Inferenz mit heruntergeladenen CosyPose Inferenzdaten --> inference_data.zip)

Generelle Ordnerstruktur für die Implementierung von Datensätzen, die einghalten werden muss

Wichtig: Die von der BOP-Challenge heruntergeladenen Datensätze müssen evtl. umbenannt werden, damit die Namenskonventionen wie hier sind

Hier am Beispiel der Datensatzes "ITODD" und "TLESS":

- data
  - bop
    - itodd
      - base
      - models
        - obj_000001.ply
        - ...
      - test
        - 000001
          - depth
          - gray
          - ...
      - train_pbr
        - 000001
          - depth
          - gray
          - ...
    - tless
      - base
      - models_cad
        - obj_000001.ply
        - ...
      - test_primesense
        - 000001
        - 000002
        - ...
      - train_pbr
        - 000001
        - 000002
        - ...
  - models
    - itodd-3qnq15p6.com
      pact.ckpt
  - surface_samples (from the inference_data.zip)
  - surface_samples_normals (from the inference_data.zip)
  - detection_results (from the inference_data.zip)
- detection_results (from the inference_data.zip) --> hier nochmal nötig für infer_debug
- surfemb
- environment.yaml
- ...

## 3. Wichtige Befehle

Testen eines Datensatzes mit CosyPose Detections:
python -m surfemb.scripts.infer_debug data/models/itodd-3qnq15p6.compact.ckpt --device cpu --real --detection

Inferenz eines Datensatzes:
python -m surfemb.scripts.infer data/models/itodd-3qnq15p6.compact.ckpt --device cpu
