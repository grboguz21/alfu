# Modül Geliştirme Kuralları

## Senin Görevin

Senden **sadece modül `.py` dosyasını** yazman isteniyor.  
`runner.py` entegrasyonu ve kamera config JSON düzenlemeleri ayrıca yapılacak — bunlara dokunma.

Teslim edeceğin tek şey: `modul_adin.py`

---

## Sistem Bağlamı

Her kamera bağımsız bir process olarak çalışır. Her process şu döngüyü sürekli tekrarlar:

```
Kameradan kare al
    → YOLO ile nesneleri tespit et
    → Tracker ile nesne ID'lerini ata
    → modules.update()  ← senin modülün burada çağrılır
    → modules.draw()    ← senin modülün burada çizim yapar
    → API'ye rapor gönder (modules.get_data() sonucunu kullanır)
```

Modülün bu döngünün içinde yaşar. Kamera açma, model yükleme pipeline'ı, RTSP okuma gibi şeylerle ilgilenmezsin.

---

## İçindekiler

1. [Modül Türleri](#1-modül-türleri)
2. [İsimlendirme](#2-i̇simlendirme)
3. [BaseModule Arayüzü](#3-basemodule-arayüzü)
4. [Tam İskelet — Buradan Başla](#4-tam-i̇skelet--buradan-başla)
5. [Sabitler Bölümü](#5-sabitler-bölümü)
6. [`__init__`](#6-__init__)
7. [Kalıcılık — `_save_state` ve `_load_state`](#7-kalıcılık--_save_state-ve-_load_state)
8. [`_check_daily_reset`](#8-_check_daily_reset)
9. [`update()`](#9-update)
10. [`get_data()`](#10-get_data)
11. [`draw()`](#11-draw)
12. [`shutdown()` ve `reset()`](#12-shutdown-ve-reset)
13. [GPU Kullanımı](#13-gpu-kullanımı)
14. [Teslim Öncesi Kontrol Listesi](#14-teslim-öncesi-kontrol-listesi)

---

## 1. Modül Türleri

Modülünü yazmadan önce türünü belirle. İkisi birbirinden farklı şablona sahip.

### Tür A — Pipeline Sonuçlarını Kullanan

Ana pipeline'ın (YOLO + tracker) ürettiği `bboxes`, `class_ids`, `object_ids` verilerini `update()` içinde kullanır. Kendi YOLO modeli **yoktur**.

Ne zaman seçilir: Bounding box konumuna, ID'sine veya sınıfına bakarak karar vereceksen.

**Mevcut örnekler:** Kapı açık/kapalı tespiti, yasak alan ihlali, depo kişi sayımı

---

### Tür B — Kendi Modelini / Algoritmasını Çalıştıran

`update()`'e gelen `bboxes` parametrelerini **tamamen yok sayar**. Her karede kendi YOLO modelini veya klasik görüntü işleme yöntemini çalıştırır.

Ne zaman seçilir: Ana pipeline'ın tespit ettiği nesneler değil, başka bir model veya piksel analizi kullanacaksan.

**Mevcut örnekler:** Çalışma süresi takibi (YOLO pose), forklift süresi (özel YOLO), lens engel tespiti (Laplacian)

> **Tür B zorunluluğu:** `GPU_LOCK` kullanmak zorundasın. Detay için [Bölüm 13](#13-gpu-kullanımı).

---

## 2. İsimlendirme

| Ne | Kural | Örnek |
|----|-------|-------|
| Dosya adı | `snake_case.py` | `bekleme_suresi.py` |
| Sınıf adı | `PascalCase` + `Module` soneki | `BeklemeSuresiModule` |
| State dosyası öneki | modülün mantıksal adı (sabit, kısa) | `state/bekleme_suresi_{name}.json` |
| Config `"type"` değeri | `snake_case`, dosya adıyla tutarlı | `"type": "bekleme_suresi"` |

---

## 3. BaseModule Arayüzü

Sınıfın şu sınıftan türetilmeli:

```python
# engine/modules/base.py — değiştirme, sadece referans için

class BaseModule:
    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        raise NotImplementedError

    def get_data(self) -> dict:
        raise NotImplementedError

    def draw(self, frame):
        return frame      # varsayılan: frame'e dokunmaz

    def reset(self):
        pass              # isteğe bağlı

    def shutdown(self):
        pass              # isteğe bağlı
```

`update()` ve `get_data()` **zorunludur** — diğerleri ihtiyaca göre override edilir.

---

## 4. Tam İskelet — Buradan Başla

Aşağıdaki iskelet her yeni modül için başlangıç noktasıdır. İhtiyacına göre doldur, gereksiz bölümleri sil.

```python
"""
Modül Adı
---------
Tek satır açıklama.

Config example:
    {
        "type":       "modul_adi",
        "name":       "modul_adi_cam1",
        "parametre1": deger,
        "parametre2": deger
    }

get_data() output:
    {
        "Some Value":   float,
        "my_alert":     bool
    }
"""

import os
import json
import time as _time
import datetime

import cv2
import numpy as np

from .base import BaseModule
# Tür B modülü ise şunları da ekle:
# import torch
# from ultralytics import YOLO
# from engine.shared_memory import GPU_LOCK

# ==================== CONFIG ====================

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30

# Görsel sabitler (draw() kullanıyorsan)
COLOR_ACTIVE = (0, 255, 0)
COLOR_IDLE   = (0, 0, 255)


# ==================== MODULE ====================

class ModulAdiModule(BaseModule):

    def __init__(self, name: str,
                 parametre1: float = 1.0,
                 parametre2: list  = None,
                 show_panel: bool  = True,
                 **_kwargs):
        self.name        = name
        self.parametre1  = parametre1
        self.parametre2  = parametre2
        self.show_panel  = show_panel

        # İç durum değişkenleri
        self._total_seconds   = 0.0
        self._is_active       = False
        self._session_start   = None
        self._should_alert    = False
        self._last_reset_date = None
        self._last_save_time  = 0.0

        # draw() için — update() çağrılmadan önce None olmalı
        self._last_status = None

        self._load_state()
        print(f"✅ ModulAdiModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"modul_adi_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date":          (self._last_reset_date.isoformat()
                                  if self._last_reset_date else None),
                "total_seconds": self._total_seconds,
            }
            tmp = self._state_path() + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            for attempt in range(5):
                try:
                    os.replace(tmp, self._state_path())
                    break
                except OSError:
                    if attempt < 4:
                        _time.sleep(0.05)
                    else:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        raise
        except Exception as e:
            print(f"[{self.name}] State save error: {e}")

    def _load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            print(f"[{self.name}] No state file, starting fresh.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            saved_date = state.get("date")
            today      = datetime.datetime.now().date().isoformat()
            if saved_date != today:
                print(f"[{self.name}] State outdated ({saved_date}), starting fresh.")
                return
            self._total_seconds   = float(state.get("total_seconds", 0.0))
            self._last_reset_date = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded ({self._total_seconds/60:.1f} min)")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._total_seconds   = 0.0
            self._is_active       = False
            self._session_start   = None
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()           # Her zaman ilk satır
        self._should_alert = False          # Her karede sıfırla
        now = _time.time()

        # — Tür A: bboxes'ı işle —
        for bbox, cls_id in zip(bboxes, class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # ... analiz mantığı ...

        # — Tür B: kendi modelini çalıştır (bboxes yok say) —
        # with GPU_LOCK, torch.no_grad():
        #     results = self._model(frame, verbose=False)
        # boxes = results[0].boxes.xyxy.cpu().numpy()   # lock dışında işle

        # draw() için son durumu sakla
        self._last_status = { "is_active": self._is_active }

        # Periyodik state kayıt
        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        now   = _time.time()
        total = self._total_seconds
        if self._is_active and self._session_start:
            total += now - self._session_start

        return {
            "Active Time Minutes": round(total / 60, 1),
            "is_active":           self._is_active,
            "my_alert":            self._should_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:   # update() henüz çağrılmadı
            return frame
        if self.show_panel:
            frame = self._draw_panel(frame)
        return frame

    def _draw_panel(self, frame):
        # Örnek panel çizimi
        h, w  = frame.shape[:2]
        color = COLOR_ACTIVE if self._is_active else COLOR_IDLE
        cv2.putText(frame, f"Status: {self._is_active}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        if self._is_active and self._session_start:
            self._total_seconds += _time.time() - self._session_start
            self._session_start  = None
            self._is_active      = False
        self._save_state()
```

---

## 5. Sabitler Bölümü

`# ==================== CONFIG ====================` altında tanımla.

**Her modülde zorunlu:**
```python
STATE_DIR         = "state"   # asla değiştirme
SAVE_INTERVAL_SEC = 30        # otomatik kayıt aralığı (saniye)
```

**Kural:** `if elapsed > 30` gibi sihirli sayıları koda gömme. Sabit olarak tanımla:
```python
ALERT_SECONDS = 30
# ...
if elapsed > ALERT_SECONDS:   # okunabilir ve değiştirilebilir
```

---

## 6. `__init__`

### İmza Kuralları

```python
def __init__(self, name: str, <zorunlu_param>, <isteğe_bağlı>=<varsayılan>, **_kwargs):
```

- `name: str` her zaman **ilk** parametre.
- `**_kwargs` her zaman **son** parametre — config'den gelen bilinmeyen alanlar hata vermez.
- Type hint yaz: `roi: list`, `threshold: float`, `enabled: bool`.

### Sıra Önemli

```
1. self.xxx = xxx      → config parametrelerini ata
2. self._model = YOLO  → Tür B ise modeli yükle
3. self._xxx = 0.0     → iç durum değişkenlerini başlat
4. self._last_xyz = None → draw() None güvenliği için
5. self._load_state()  → en sonda, her zaman
6. print("✅ ...")     → hazır mesajı
```

**Yasak:** `cv2.VideoCapture`, ağ bağlantısı veya blocking operasyon açma.

---

## 7. Kalıcılık — `_save_state` ve `_load_state`

### Neden Atomik Yazma?

`os.replace()` POSIX atomik operasyondur — dosya hiçbir zaman yarı yazılmış halde kalmaz. `.{pid}.tmp` suffix'i aynı anda çalışan birden fazla kamera process'inin birbirine yazmasını önler. Bu kalıbı birebir kullan, değiştirme.

### `_state_path()`

```python
def _state_path(self) -> str:
    return os.path.join(STATE_DIR, f"<prefix>_{self.name}.json")
    # Örnek: "state/bekleme_suresi_cam2.json"
```

`<prefix>` — modülün kısa mantıksal adı. Her modül farklı prefix kullanır, çakışma olmaz.

### `_save_state()` — Değiştirmeden Kopyala

```python
def _save_state(self):
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        state = {
            "date":          (self._last_reset_date.isoformat()
                              if self._last_reset_date else None),
            "total_seconds": self._total_seconds,
            # kaydetmek istediğin diğer alanlar
        }
        tmp = self._state_path() + f".{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        for attempt in range(5):
            try:
                os.replace(tmp, self._state_path())
                break
            except OSError:
                if attempt < 4:
                    _time.sleep(0.05)
                else:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    raise
    except Exception as e:
        print(f"[{self.name}] State save error: {e}")
```

### `_load_state()` Kuralları

- `state.get("alan", varsayılan)` — her alan için varsayılan değer ver.
- `saved_date != today` ise **hiçbir şey yükleme**, `return` ile çık.
- Her şeyi `try/except` içine al — bozuk dosya programı çökertmemeli.

---

## 8. `_check_daily_reset`

Veri biriktiren her modülde zorunludur. `update()` içinde **her karede ilk satırda** çağrılır.

```python
def _check_daily_reset(self):
    today = datetime.datetime.now().date()
    if self._last_reset_date is None:
        self._last_reset_date = today
        return
    if today != self._last_reset_date:
        self._total_seconds   = 0.0    # sıfırlanacak tüm sayaçlar
        self._is_active       = False
        self._session_start   = None
        self._last_reset_date = today
        self._save_state()
        print(f"[{self.name}] Daily reset → {today}")
```

**Akış:**
```
update() çağrısı
    └── _check_daily_reset()
            ├── ilk çağrı → bugünü kaydet, çık
            └── gün değişti → sayaçları sıfırla + kaydet
```

---

## 9. `update()`

### İmza — Hiç Değiştirme

```python
def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
```

### Parametreler

| Parametre | Tür | İçerik |
|-----------|-----|--------|
| `bboxes` | `np.ndarray` veya `[]` | Tespit kutuları, her satır `[x1, y1, x2, y2]` |
| `class_ids` | `np.ndarray` veya `[]` | Her bbox için YOLO sınıf indeksi (int) |
| `scores` | `np.ndarray` veya `[]` | Her bbox için güven skoru, 0–1 arası |
| `object_ids` | `np.ndarray`, `[]` veya `None` | Tracker'ın atadığı kalıcı ID'ler |
| `frame` | `np.ndarray` | BGR görüntü |
| `class_names` | `dict` | `{0: "person", 2: "car", ...}` |

### Kritik Kurallar

**`frame`'i `update()` içinde değiştirme.** Çizim sadece `draw()` içinde yapılır.

**`bboxes` boş gelebilir** — kontrol etmeden zip'leme:
```python
if len(bboxes) == 0:
    # tespit yok durumu
    ...
for bbox, cls_id in zip(bboxes, class_ids):
    ...
```

**`object_ids` None gelebilir** — doğrudan erişme:
```python
if object_ids is not None:
    obj_id = object_ids[i]
```

**`draw()` için son durumu sakla:**
```python
self._last_status = {"is_active": self._is_active, "count": self._count}
```
`draw()`, `update()`'ten hemen sonra çağrılır; o anki durumu `self._last_xyz` attribute'larında bırakman gerekir.

**`update()` hiçbir şey döndürmez.**

**Periyodik kayıt her `update()` içinde olmalı:**
```python
if now - self._last_save_time >= SAVE_INTERVAL_SEC:
    self._save_state()
    self._last_save_time = now
```

### Tür A Şablonu

```python
def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
    self._check_daily_reset()
    self._should_alert = False
    now = _time.time()

    for bbox, cls_id in zip(bboxes, class_ids):
        if int(cls_id) != self._target_class_id:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # ... analiz mantığı ...

    self._last_status = {"is_active": self._is_active}

    if now - self._last_save_time >= SAVE_INTERVAL_SEC:
        self._save_state()
        self._last_save_time = now
```

### Tür B Şablonu

```python
def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
    # bboxes tamamen yok sayılır
    self._check_daily_reset()
    now = _time.time()

    with GPU_LOCK, torch.no_grad():
        results = self._model(frame, verbose=False)

    # lock DIŞINDA işle
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else []
    # ... analiz mantığı ...

    self._last_status = {"is_active": self._is_active}

    if now - self._last_save_time >= SAVE_INTERVAL_SEC:
        self._save_state()
        self._last_save_time = now
```

---

## 10. `get_data()`

### İmza

```python
def get_data(self) -> dict:
```

### Kurallar

**Düz sözlük döndür.** İç içe dict, liste veya nesne kabul edilmez — API bu çıktıyı doğrudan alır.

**Anahtarlar İngilizce ve okunabilir olmalı.** API'ye ve rapora bu isimler gider:
```python
# Doğru
"Total Open Seconds", "Forklift Working Time", "Person Count"

# Yanlış
"total_sec", "fw_t", "cnt"
```

**Yan etkisi olmamalı.** `get_data()` çağrısı iç durumu değiştirmez, hiçbir sayacı sıfırlamaz.

**Alarm bayrağı taşıyorsa `_alert` ile bitir:**
```python
"open_duration_alert": True   # main.py bunu alarm tetiklemek için izler
"should_alert":        True
```

**Açık oturumu hesaba kat:**
```python
def get_data(self) -> dict:
    now   = _time.time()
    total = self._total_seconds
    if self._is_active and self._session_start:
        total += now - self._session_start  # o anki oturum dahil

    return {
        "Active Time Minutes": round(total / 60, 1),
        "is_active":           self._is_active,
        "my_alert":            self._should_alert,
    }
```

---

## 11. `draw()`

### İmza

```python
def draw(self, frame):
    ...
    return frame      # her durumda frame döndür
```

### Kurallar

**Her zaman `frame` döndür.** `ModuleRunner` dönüş değerini bir sonraki modüle geçirir; `None` dönerse sistem çöker.

**İlk satırda None kontrolü yap:**
```python
def draw(self, frame):
    if self._last_status is None:   # update() henüz hiç çağrılmadı
        return frame
    # ...
    return frame
```

**`frame`'i doğrudan değiştir; kopya alma.** `frame.copy()` pahalıdır.

**Şeffaf panel için:**
```python
overlay = frame.copy()
cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
```

**`show_panel` parametresine uyu:**
```python
def draw(self, frame):
    if self._last_status is None:
        return frame
    # ROI/polygon her zaman çiz
    frame = self._draw_zones(frame)
    # Panel sadece show_panel=True ise
    if self.show_panel:
        frame = self._draw_panel(frame)
    return frame
```

---

## 12. `shutdown()` ve `reset()`

### `shutdown()` — Veri Kaydeden Modüllerde Zorunlu

Sistem kapanırken `ModuleRunner` her modülün `shutdown()`'ını çağırır. Açık oturumu kapat ve diske kaydet:

```python
def shutdown(self):
    if self._is_active and self._session_start:
        self._total_seconds += _time.time() - self._session_start
        self._session_start  = None
        self._is_active      = False
    self._save_state()
```

### `reset()` — İsteğe Bağlı

Sayaçların dışarıdan manuel sıfırlanması gerekirse uygula. Günlük sıfırlama için `reset()` değil `_check_daily_reset()` kullanılır.

```python
def reset(self):
    self._total_seconds = 0.0
    self._is_active     = False
    self._session_start = None
```

---

## 13. GPU Kullanımı

### Kural: Tür B Modüllerde `GPU_LOCK` Zorunlu

```python
from engine.shared_memory import GPU_LOCK
import torch

# update() içinde:
with GPU_LOCK, torch.no_grad():
    results = self._model(frame, verbose=False)
```

**Neden?** Birden fazla kamera process'i aynı GPU'yu paylaşır. `GPU_LOCK` aynı anda yalnızca bir modelin GPU'ya erişmesini sağlar; aksi hâlde CUDA bellek taşması olur.

### Lock Bloğunu Kısa Tut

```python
# DOĞRU — lock içinde sadece inference
with GPU_LOCK, torch.no_grad():
    results = self._model(frame, verbose=False)
boxes = results[0].boxes.xyxy.cpu().numpy()   # lock dışında

# YANLIŞ — lock içinde döngü ve işlem
with GPU_LOCK, torch.no_grad():
    results = self._model(frame, verbose=False)
    boxes   = results[0].boxes.xyxy.cpu().numpy()
    for box in boxes:          # gereksiz yere lock içinde
        ...
```

**Tür A modülleri (kendi modeli olmayanlar) `GPU_LOCK` kullanmaz.**

---

## 14. Teslim Öncesi Kontrol Listesi

Dosyayı göndermeden önce şunları kontrol et:

**Yapı**
- [ ] Docstring'de `Config example:` ve `get_data() output:` var mı?
- [ ] `BaseModule`'dan türetiliyor mu? (`class XxxModule(BaseModule):`)
- [ ] `from .base import BaseModule` import'u var mı?
- [ ] Tür B ise `from engine.shared_memory import GPU_LOCK` import'u var mı?

**`__init__`**
- [ ] İlk parametre `name: str` mi?
- [ ] Son parametre `**_kwargs` mi?
- [ ] `_load_state()` en sonda çağrılıyor mu?
- [ ] `draw()` için kullanılan değişkenler `None` ile başlatıldı mı?

**Kalıcılık**
- [ ] `STATE_DIR = "state"` tanımlı mı?
- [ ] `SAVE_INTERVAL_SEC = 30` tanımlı mı?
- [ ] `_state_path()` doğru prefix kullanıyor mu?
- [ ] `_save_state()` atomik yazma (`.tmp` + `os.replace`) kullanıyor mu?
- [ ] `_load_state()` tarih kontrolü yapıyor mu? Eski tarihte `return` ile çıkıyor mu?
- [ ] Her `state.get("alan", varsayılan)` varsayılan değere sahip mi?

**`update()`**
- [ ] İmza tam olarak `(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict)` mi?
- [ ] İlk satır `self._check_daily_reset()` mi?
- [ ] `frame` yerinde değiştirilmiyor mu?
- [ ] `bboxes` boş liste durumu kontrol edildi mi?
- [ ] `object_ids` None durumu kontrol edildi mi?
- [ ] Tür B ise `GPU_LOCK` + `torch.no_grad()` kullanılıyor mu?
- [ ] `draw()` için `self._last_xyz` güncelleniyor mu?
- [ ] Periyodik `_save_state()` çağrısı var mı?
- [ ] Dönüş değeri yok mu?

**`get_data()`**
- [ ] Düz sözlük döndürüyor mu (iç içe yok)?
- [ ] Anahtarlar İngilizce ve okunabilir mi?
- [ ] Yan etkisi yok mu (sayaç sıfırlamıyor)?
- [ ] Açık oturum varsa toplama dahil ediyor mu?

**`draw()`**
- [ ] Her dalda `frame` döndürüyor mu?
- [ ] İlk satırda `self._last_xyz is None` kontrolü var mı?
- [ ] `show_panel` flag'ine uyuyor mu?

**`shutdown()`**
- [ ] Veri topluyorsa açık oturumu kapatıp `_save_state()` çağırıyor mu?
