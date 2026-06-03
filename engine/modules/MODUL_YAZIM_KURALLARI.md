# Modül Yazım Kuralları

## 1. İki Modül Tipi Vardır

### Tip A — Sistem Tracker'ını Kullanan
Sistemin YOLO + OCSoft çıktısını alır. Kendi inference'ı yoktur.
`update()` parametrelerindeki `bboxes, class_ids, object_ids` kullanılır.

**Ne zaman:** Araç sayma, istasyon doluluk, kişi takibi gibi
sistem modelinin zaten algıladığı nesneler üzerinde analiz yapılacaksa.

**Örnek:** `station_occupancy.py`

```python
def update(self, bboxes, class_ids, scores, object_ids, frame, class_names):
    for bbox, cid, oid in zip(bboxes, class_ids, object_ids):
        cls_name = class_names.get(int(cid), "")
        # bboxes üzerinde çalış
```

---

### Tip B — Kendi Modeli Olan
`__init__`'te ayrı bir YOLO modeli yükler. `update()` içinde
`frame` üzerinde kendi inference'ını çalıştırır. `bboxes` parametrelerini kullanmaz.

**Ne zaman:** Sistemin algılayamadığı özel sınıflar gerekiyorsa
(nakit para, yangın, özel ürün vb.) ve bunun için ayrı eğitilmiş
bir model varsa.

**Örnek:** `nakit_module.py`

```python
def __init__(self, name, model_path, ...):
    self._model = YOLO(model_path)   # kendi modeli

def update(self, bboxes, class_ids, scores, object_ids, frame, class_names):
    results = self._model(frame, ...)  # kendi inference
    # bboxes parametresi KULLANILMAZ
```

---

## 2. Zorunlu Kurallar

### 2.1 BaseModule'dan türet
```python
from .base import BaseModule

class BenimModulum(BaseModule):
    ...
```

### 2.2 `__init__` parametreleri config'den gelir
Config'de yazılan her alan (type hariç) direkt `__init__`'e parametre olarak aktarılır.
Parametre adları config key'leriyle birebir eşleşmeli.

```json
{"type": "benim_modulom", "name": "test", "esik": 0.7}
```
```python
def __init__(self, name: str, esik: float = 0.5):
```

### 2.3 `update()` imzası değişmez
```python
def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
```
Kullanmadığın parametreleri yoksay, imzayı bozma.

### 2.4 `get_data()` düz dict döndürür
```python
def get_data(self) -> dict:
    return {"anahtar": deger}   # int, float, bool, str
```
İç içe dict veya liste döndürme — API bunu beklemez.

### 2.5 State instance variable'da tutulur
Global değişken kullanma. Her kamera bağımsız bir instance'tır.
```python
# ❌ yanlış
count = 0

# ✅ doğru
self._count = 0
```

### 2.6 `draw()` frame alır, frame döndürür
```python
def draw(self, frame):
    # frame üzerinde cv2 çizimi yap
    return frame   # mutlaka döndür
```

---

## 3. Runner'a Kayıt

`engine/modules/runner.py`:
```python
from .benim_modulom import BenimModulum

MODULE_REGISTRY = {
    ...
    "benim_modulom": BenimModulum,
}
```

---

## 4. Config'e Ekle

```json
"modules": [
    {
        "type":       "benim_modulom",
        "name":       "sonuc_adi",
        "model_path": "models/ozel_model.pt",
        "zone":       [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    }
]
```

---

## 5. Tip B için Ek Kurallar (Kendi Modeli Olan)

- Model `__init__`'te bir kez yüklenir, her frame'de tekrar yüklenmez
- `update()` içinde sadece `frame` kullanılır, `bboxes` yoksayılır
- `detection_classes` config alanı bu modülü etkilemez (sistem modelini filtreler)
- Kendi modelinin sınıf filtresi `__init__`'te parametre olarak alınır

```python
def __init__(self, name, model_path, classes=None, confidence=0.5):
    self._model      = YOLO(model_path)
    self._classes    = classes       # ["nakit", "kart"] gibi
    self._confidence = confidence
```

---

## 6. Alarm Gönderme

Modül doğrudan alarm gönderemez. Alarm mantığı `get_data()` içinde
bir flag ile dışarıya taşınır, `main.py`'de `send_alarm()` çağrılır.

```python
# Modül içinde:
def get_data(self):
    return {
        "nakit_var": self._detected,   # True/False
        "adet":      self._count
    }
```

```python
# main.py içinde (özel alarm gerekiyorsa):
data = modules.get_data()
if data.get("nakit_var"):
    report.send_alarm("nakit_alarm", data=data, media_path=minio_path)
```
