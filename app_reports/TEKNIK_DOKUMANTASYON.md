# OPET Sistemi — Kapsamlı Teknik Dokümantasyon

## İngiltere Market / YourEye Projesi

---

## 1. PROJE GENEL BAKIŞ

**Proje Adı:** OPET (Gerçek Zamanlı Video Analiz Sistemi)  
**İşletici Firma:** YourEye  
**Çalışma Dizini:** `c:\Youreye Projects\ingiltere_market\kafkas\`  
**Ana Amaç:** IP kameralardan gerçek zamanlı nesne tespiti, takibi ve iş analizi yapmak  
**Dil:** Python 3  
**Durum:** Aktif üretim sistemi  
**API Endpoint:** `https://app.youreye.co.uk/api`

Bu sistem, İngiltere'deki bir market/depo lokasyonundaki 6 kamerayı gerçek zamanlı izler; kapı durumu, çalışma süresi, forklift hareketi ve alan ihlali gibi analizleri otomatik olarak yapar ve raporları bulut sistemine gönderir.

---

## 2. PROJE DİZİN YAPISI

```
kafkas/
├── main.py                        ← Giriş noktası (multiprocessing orkestratörü)
├── run.py                         ← Log sarmalayıcı + otomatik yeniden başlatma
├── report_manager.py              ← API iletişimi ve kuyruk yönetimi
├── config.json                    ← Ana yapılandırma dosyası
├── README.md                      ← Hızlı başlangıç rehberi
├── SISTEM_DOKUMANI.md
├── PROJE_ANALİZİ.md
├── BUGLAR_VE_COZUMLER.md
├── GELISTIRICI_KURALLARI.md
├── KAMERA_HAREKETI_SPEC.md
├── KAPANIŞ_RAPORU_SPEC.md
├── KULLANIM_KILAVUZU.md
├── api_test.py                    ← API bağlantı testi
├── is_software_alive.py           ← Sağlık kontrolü scripti
├── minio_test.py                  ← MinIO bağlantı testi
├── make_reference.py              ← Referans görüntü oluşturucu
├── rescale_rois.py                ← ROI koordinat ölçeklendirici
├── roi_ciz.py                     ← ROI çizim aracı
│
├── cameras/                       ← Kamera yapılandırmaları
│   ├── ingiltere-kamera-1.json
│   ├── ingiltere-kamera-2.json
│   ├── ingiltere-kamera-3.json
│   ├── ingiltere-kamera-4.json
│   ├── ingiltere-kamera-5.json
│   └── ingiltere-kamera-6.json
│
├── state/                         ← Kalıcı durum dosyaları
│   ├── door_open_closed_*.json
│   ├── calisma_suresi_*.json
│   ├── forklift_*.json
│   ├── health_*.json
│   └── lens_detector_state.json
│
└── engine/                        ← Çekirdek işleme motoru
    ├── multithreading_tracking.py ← 3-thread orkestratörü
    ├── object_detection.py        ← YOLO toplu çıkarım motoru
    ├── object_tracking.py         ← İzleyici fabrikası
    ├── minio_manager.py           ← MinIO yükleme yöneticisi
    ├── shared_memory.py           ← Thread-güvenli kuyruklar
    ├── rtsp_probe.py              ← GStreamer boru hattı oluşturucu
    ├── fps.py                     ← FPS hesaplama yardımcısı
    ├── camera.py                  ← Kamera soyutlama sınıfı
    ├── modules/                   ← Analiz modülleri
    │   ├── __init__.py
    │   ├── base.py                ← BaseModule arayüzü
    │   ├── runner.py              ← ModuleRunner orkestratörü
    │   ├── calisma_suresi.py      ← Çalışma süresi takibi
    │   ├── door_open_closed.py    ← Kapı açık/kapalı tespiti
    │   ├── door_report.py         ← Günlük kapı raporu
    │   ├── forklift_module.py     ← Forklift çalışma süresi
    │   ├── forklift_tracker.py    ← Forklift hareket takibi
    │   ├── face_blurring.py       ← Yüz anonimleştirme
    │   ├── lens_detector.py       ← Lens engel tespiti
    │   ├── restricted_zone_module.py ← Yasak alan tespiti
    │   ├── warehouse_person_counter.py ← Depo kişi sayımı
    │   └── hardware_monitor.py    ← GPU/CPU izleme
    └── trackers/                  ← İzleme algoritmaları
        ├── ocsort/                ← OCSoft (varsayılan)
        ├── strongsort/            ← StrongSORT (alternatif)
        └── bytetrack/             ← ByteTrack (devre dışı)
```

---

## 3. MİMARİ GENEL GÖRÜNÜM

Sistem üç katmandan oluşur:

```
┌─────────────────────────────────────────────────────────────┐
│               KATMAN 1: GIRIŞ NOKTASI                        │
│         main.py → run.py (üretim sarmalayıcısı)             │
└─────────────────────┬───────────────────────────────────────┘
                      │ multiprocessing.Pool
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   [Process 1]   [Process 2]   [Process N]
   Kamera-1      Kamera-2      Kamera-6

┌─────────────────────────────────────────────────────────────┐
│              KATMAN 2: KAMERA IŞLEME (Her process)           │
│                                                              │
│  Thread 1: Cap (Kare okuma)                                  │
│     └──→ cap_buffer (Queue, max:4)                           │
│  Thread 2: Detection+Tracking (YOLO + OCSoft)                │
│     └──→ tracking_buffer (Queue, max:4)                      │
│  Ana Thread: Modül analizi + Raporlama                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────────┐
│ KATMAN 3:     │           │ KATMAN 3:         │
│ MinIO         │           │ YourEye Backend   │
│ (S3 Depolama) │           │ API               │
└───────────────┘           └───────────────────┘
```

---

## 4. YAPILANDIRMA SİSTEMİ

### 4.1 Ana Yapılandırma — `config.json`

```json
{
    "hardware_monitor": {
        "enabled": true,
        "gpu_index": 0,
        "log_interval_sec": 10
    },
    "minio": {
        "endpoint": "localhost:9000",
        "bucket_name": "ai-outputs",
        "branch_name": "kafkas",
        "secure": false
    },
    "api": {
        "gateway_base": "https://app.youreye.co.uk/api",
        "api_key": "yourEYE060734",
        "branch_id": "8475c722-aab6-4b33-9786-a7e2c213759e"
    }
}
```

### 4.2 Kamera Yapılandırma Şeması

Her kamera JSON dosyası şu alanları içerir:

| Alan | Tür | Açıklama |
|------|-----|----------|
| `enabled` | bool | Kameranın aktif olup olmadığı |
| `minio_folder` | string | MinIO'daki klasör adı |
| `camera_id` | UUID | API'deki kamera kimliği |
| `module_id` | UUID | API'deki modül kimliği |
| `rtsp_url` | string | Kamera RTSP adresi |
| `video_path` | string | Yerel video dosyası (test için) |
| `engine_path` | string | YOLO model dosyası (boş = tespit yok) |
| `batch_size` | int | YOLO toplu çıkarım boyutu |
| `detection_fps` | float | Saniyede maksimum tespit sayısı |
| `resize_factor` | float | Kare boyutu çarpanı (0.3 = %30) |
| `modules` | array | Aktif analiz modülleri listesi |
| `alarms` | array | Tetiklenecek alarm tanımları |
| `reports` | array | Periyodik rapor tanımları |

### 4.3 Kamera Envanteri

| Kamera | Klasör | Tip | Ana Modül | Motor |
|--------|--------|-----|-----------|-------|
| Kamera-1 | ingiltere-kamera-1 | Kapı İzleme | door_open_closed | yolov8n.pt |
| Kamera-2 | ingiltere-kamera-2 | Çalışma Alanı (4 bölge) | calisma_suresi | — |
| Kamera-3 | ingiltere-kamera-3 | Çalışma Alanı (1 bölge) | calisma_suresi | — |
| Kamera-4 | ingiltere-kamera-4 | Forklift Takibi | forklift_suresi | models/forklift_best.pt |
| Kamera-5 | ingiltere-kamera-5 | Yapılandırılmış | — | — |
| Kamera-6 | ingiltere-kamera-6 | Yapılandırılmış | — | — |

---

## 5. KARE İŞLEME AKIŞI

Her kamera için şu pipeline çalışır:

```
RTSP / Video Kaynağı
        │
        ▼
┌─────────────────────┐
│    Thread 1: Cap    │  cv2.VideoCapture veya GStreamer
│  • Kare okuma       │  Maks. 30 FPS
│  • Otomatik yeniden │
│    bağlantı         │
│  • Boyut küçültme   │
│    (resize_factor)  │
└──────────┬──────────┘
           │ cap_buffer (Queue, max:4)
           ▼
┌──────────────────────────────────┐
│   Thread 2: Tespit + Takip       │
│  • YOLO toplu çıkarım            │
│  • OCSoft Kalman filtresi        │
│  • Nesne ID atama                │
│  • FPS sınırlaması (10 FPS)      │
└──────────┬───────────────────────┘
           │ tracking_buffer (Queue, max:4)
           ▼
┌──────────────────────────────────────────┐
│        Ana Thread: Analiz & Raporlama    │
│  1. modules.update() → tüm modüller      │
│  2. modules.draw() → görselleştirme      │
│  3. report.check_reports() → zamanlama   │
│  4. minio.upload() → görüntü yükleme    │
│  5. report.send_report/alarm() → API     │
└──────────────────────────────────────────┘
```

---

## 6. MODÜL SİSTEMİ

### 6.1 BaseModule Arayüzü — `engine/modules/base.py`

Tüm modüller dört metodu uygular:

```python
class BaseModule:
    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names):
        """Her karede izleme sonuçlarıyla çağrılır"""

    def get_data(self) -> dict:
        """Rapor tetiklendiğinde API için veri döner"""

    def draw(self, frame):
        """İsteğe bağlı — açıklamalı kare döner"""

    def shutdown(self):
        """İsteğe bağlı — kapanışta durumu diske kaydeder"""
```

### 6.2 ModuleRunner — `engine/modules/runner.py`

Tüm modülleri yönetir:
- Yapılandırmadan modülleri yükler ve başlatır
- Her karede `update()` çağrısı yapar
- Rapor zamanında `get_data()` toplar
- Görselleştirme için `draw()` çağrısı yapar
- `disabled_modules` listesini takip eder

### 6.3 Uygulanan Modüller (9 Modül)

---

#### Modül 1: `door_open_closed` — Kapı Durumu Tespiti

**Dosya:** `engine/modules/door_open_closed.py`

**Algoritma:**
- ROI bölgesindeki piksel standart sapmasını hesaplar
- Düşük standart sapma → Kapalı (düzgün yüzey)
- Yüksek standart sapma → Açık (çeşitli arka plan)

**Durum Makinesi:**
```
KAPALI ──────────────────────────────── AÇIK
  │    consensus_seconds süre geçince    │
  └──────────────────────────────────────┘
```

**Kişi Takibi:** Polygon tabanlı kişi tespiti

**Alarm:** Kapı açık + `alert_open_seconds` süre + kişi yok → alarm

**Çıktı Örneği:**
```json
{
    "Total Open Seconds": 240,
    "No Person Seconds": 60,
    "open_duration_alert": true
}
```

**Kullanılan kamera:** Kamera-1

---

#### Modül 2: `calisma_suresi` — Çalışma Süresi Takibi

**Dosya:** `engine/modules/calisma_suresi.py`

**Model:** YOLO pose (17 vücut noktası/kişi)

**Mantık:**
- Polygon bölgelerde kişi varlığını takip eder
- `is_special: false` → Kritik keypoint bölge içinde olmalı
- `is_special: true` → Bbox merkezi VEYA ayak VEYA herhangi keypoint + kilit mekanizması
- Gizlenme toleransı: 5 saniye

**Günlük Sıfırlama:** Gece yarısında sıfırlar, JSON'a kaydeder

**Çıktı Örneği:**
```json
{
    "Area 1": 1240,
    "Area 2": 560,
    "Area 3": 0,
    "Total Minutes": 30.33
}
```

**Kullanılan kameralar:** Kamera-2 (4 bölge), Kamera-3 (1 bölge)

---

#### Modül 3: `forklift_suresi` — Forklift Çalışma Süresi

**Dosya:** `engine/modules/forklift_module.py`

**Model:** Özel YOLO forklift dedektörü (`models/forklift_best.pt`)

**Mantık:**
- Forklifti tespit eder
- Hareket vs titreşim ayrımı (movement_tolerance: 120px)
- 10 saniye hareketsizlik → IDLE durumu
- Görüntü dışı → ÇALIŞIYOR sayılır (yapılandırılabilir)

**Çıktı Örneği:**
```json
{
    "Forklift Working Time": 1240.5
}
```

**Kullanılan kamera:** Kamera-4

---

#### Modül 4: `face_blurring` — Yüz Anonimleştirme

**Dosya:** `engine/modules/face_blurring.py`

**Model:** YOLO yüz dedektörü (`models/yolov8l_100e.pt`)

**Amaç:** Gizlilik — karelerdeki yüzleri bulanıklaştırır

**Performans:** `process_every_n` parametresiyle kare atlama

**Kullanılan kameralar:** Kamera-1, 2, 3, 4

---

#### Modül 5: `lens_detector` — Lens Engel Tespiti

**Dosya:** `engine/modules/lens_detector.py`

**Yöntem:** Laplacian bulanıklık tespiti + gece eşiği

**Eşikler:**
- `blur_threshold`: 100.0
- `night_threshold`: 30.0 (karanlık için yanlış pozitif engellemesi)

**Alarm:** `lens_closed_alarm` — oturum başına bir kez (cooldown: 999999s)

---

#### Modül 6: `door_report` — Günlük Kapı İstatistikleri

**Dosya:** `engine/modules/door_report.py`

**Amaç:** Günlük kapı açma/kapama sayımlarını toplar

**Bağımlılık:** `door_open_closed` modülünün durum verisine dayanır

---

#### Modül 7: `hardware_monitor` — Donanım İzleme

**Dosya:** `engine/modules/hardware_monitor.py`

**Amaç:** GPU kullanımı ve sıcaklığı izler

**Varsayılan aralık:** 10 saniye

---

#### Modül 8: `restricted_zone_module` — Yasak Alan Tespiti

**Dosya:** `engine/modules/restricted_zone_module.py`

**Amaç:** Kişi/nesne tanımlı polygon alanlara girerse alarm tetikler

**Alarm:** `restricted_zone_alert`

---

#### Modül 9: `warehouse_person_counter` — Depo Kişi Sayımı

**Dosya:** `engine/modules/warehouse_person_counter.py`

**Amaç:** Depo doluluk takibi

**Bölgeler:**
- `candidate_zone` — giriş bölgesi
- `definite_zone` — onaylı bölge

---

## 7. ÇEKİRDEK MOTOR MODÜLLERİ

### 7.1 `main.py` (549 satır) — Orkestratör

**Başlangıç Akışı:**
1. `config.json` + `cameras/*.json` yükler
2. Donanım izleme thread'ini başlatır (isteğe bağlı)
3. Her kamera için `multiprocessing.Pool` sürecini başlatır
4. Her süreç `track_video(**camera_config)` çalıştırır
5. Süpervizör thread ölü süreçleri yeniden başlatır (maks. 10 deneme)

**Kalp Atışı Sistemi:**
- Her 60 saniyede genel ping
- Tüm kameralar bağlıysa "sağlıklı" pingi
- Bağlantı bekleme: başlangıçta maks. 120 saniye
- Günlük: `logs/camera_health.log`

**Kritik Fonksiyonlar:**

| Fonksiyon | Amaç |
|-----------|------|
| `load_config()` | Ana + kamera yapılandırmalarını birleştirir |
| `track_video()` | Tek kamera işleme döngüsü (sonsuz) |
| `write_camera_health()` | `state/health_*.json` yazar |
| `send_heartbeat_direct()` | API'ye sağlık pingu |
| `send_camera_health_ping()` | Tüm kameraların durumu |
| `startup_pings()` | Tüm kameraların bağlanmasını bekler |

---

### 7.2 `engine/multithreading_tracking.py` (237 satır) — Thread Orkestrasyonu

**Sınıf: `MultiThreadingTracker`**

3 ayrı thread'i koordine eder:

```
┌─────────────────────────────────────────┐
│ MultiThreadingTracker                   │
│                                         │
│  start_cap_thread()       → Thread 1   │
│  start_detection_thread() → Thread 2   │
│  start_tracking_thread()  → Thread 2'ye│
│                              entegre    │
│  get_frame()              → Ana thread │
└─────────────────────────────────────────┘
```

**Tampon akışı:**
```
Cap → cap_buffer → Detection → tracking_buffer → Ana Thread
```

---

### 7.3 `engine/object_detection.py` (211 satır) — YOLO Motoru

**Sınıf: `ObjectDetection`**

| Özellik | Detay |
|---------|-------|
| Model Yükleme | YOLO (örn. `yolov8n.pt`) |
| Çıkarım | Toplu (asenkron) |
| FPS Sınırlama | `detection_fps` parametresiyle |
| CUDA Hatası | `_reload_model()` ile kurtarma |

**Toplu İşleme:**
- `batch_size` kadar kare toplar
- Son toplam az ise doldurur (padding)
- Sonuçları `detection_buffer`'a kuyruğa ekler

---

### 7.4 `engine/object_tracking.py` (79 satır) — İzleyici Fabrikası

**Sınıf: `MultiObjectTracking`**

| İzleyici | Algoritma | Durum |
|----------|-----------|-------|
| **OCSoft** | Kalman + GIoU | **Varsayılan (aktif)** |
| **StrongSORT** | Kalman + Re-ID (CNN) | Alternatif |
| **ByteTrack** | Düşük güven takibi | Devre dışı |

---

### 7.5 `engine/minio_manager.py` (203 satır) — Bulut Depolama

**Sınıf: `MinIOManager`**

**Bağlantı:** Endpoint `localhost:9000`, 5s bağlantı zaman aşımı

**Yükleme Yolları:**
```
{branch_name} / {GG-AA-YYYY} / {klasör} / {dosyaadı}
kafkas / 01-06-2026 / ingiltere-kamera-1 / snapshot.jpg
```

**Yöntemler:**

| Yöntem | Dosya Adı | Kullanım |
|--------|-----------|----------|
| `upload_report()` | `snapshot.jpg` (üzerine yazar) | Periyodik rapor |
| `upload_alert()` | `{prefix}_SS-DD-SS.jpg` (benzersiz) | Olay alarmı |
| `upload_image()` | Genel | Genel kullanım |

---

### 7.6 `report_manager.py` (488 satır) — API İletişimi

**Sınıf: `ReportManager`**

**İki Mod:**

**1. Periyodik Raporlar (zaman tabanlı):**
- Varsayılan aralık: 60 saniye
- Veri: Modül çıktıları (`get_data()`)
- Medya: MinIO snapshot
- Endpoint: `POST {gateway_base}/ai/input`

**2. Alarmlar (olay tabanlı):**
- Soğuma süresi: 5–30 saniye (yapılandırılabilir)
- Günde bir kez modu: isteğe bağlı
- Medya: Zaman damgalı alert snapshot
- Endpoint: `POST {gateway_base}/ai/alarm`

**API Yük Yapısı:**
```json
{
    "cameraId": "UUID",
    "moduleId": "UUID",
    "branchId": "UUID",
    "triggeredAt": "2026-06-01T10:30:00Z",
    "mediaFolderPath": "kafkas/01-06-2026/ingiltere-kamera-1/snapshot.jpg",
    "data": { "..." : "..." },
    "message": null
}
```

**Kuyruk Yönetimi:**
- Arka plan gönderici thread'i
- Çevrimdışı kuyruk kalıcılığı (`queue_log_*.json`)
- Başarısız istekleri yeniden dener (5 saniye bekler)
- Maks. 500 öğe (eskisi düşer)

---

### 7.7 `engine/shared_memory.py` (17 satır) — Thread-Güvenli Tamponlar

```python
GPU_LOCK = threading.Lock()              # GPU kaynağına erişim kilidi
cap_buffer = Queue(maxsize=4)            # Ham kareler
detection_buffer = Queue(maxsize=4)      # YOLO tespit sonuçları
tracking_buffer = Queue(maxsize=4)       # Takip sonuçları (ID'li)
current_batch_buffer = Queue(maxsize=4)  # Geri dönüş tamponu
```

---

## 8. İZLEME ALGORİTMALARI

### OCSoft (Varsayılan)

**Dizin:** `engine/trackers/ocsort/`

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `det_thresh` | 0 | Tespit güven eşiği |
| `iou_thresh` | 0.22137 | IoU eşleşme eşiği |
| `max_age` | 50 | Tespit olmadan track tutma (kare) |
| `min_hits` | 1 | Track aktif olmadan önce minimum hit |
| `asso_func` | giou | Eşleşme fonksiyonu |
| `use_byte` | false | ByteTrack modu |

### StrongSORT (Alternatif)

**Dizin:** `engine/trackers/strongsort/`

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `max_age` | 40 | Track tutma süresi |
| `max_dist` | 0.1594 | Re-ID mesafe eşiği |
| `nn_budget` | 100 | Özellik galerisi boyutu |

15+ önceden eğitilmiş Re-ID mimarisini destekler (OSNet, DenseNet, ResNet vb.)

### ByteTrack (Devre Dışı)

**Dizin:** `engine/trackers/bytetrack/`

Kodda yorumlanmış, kullanılmıyor.

---

## 9. ALARM VE RAPORLAMA SİSTEMİ

### 9.1 Alarm Türleri

**Sistem Alarmları (`main.py` hardcoded):**

| Alarm | Tetikleyici | Soğuma |
|-------|-------------|--------|
| `stream_kesintisi` | Akış kesilmesi | 60s |
| `camera_move` | Kamera açısı değişimi | 5s |

**Modül Alarmları (yapılandırma tabanlı):**

| Alarm | Tetikleyici Alan | Soğuma | Modül |
|-------|------------------|--------|-------|
| `kapida_kisi_yok` | `open_duration_alert` | 25s | door_open_closed |
| `lens_closed_alarm` | `should_alert` | 999999s | lens_detector |
| `restricted_zone_alert` | — | yapılandırılabilir | restricted_zone |

### 9.2 Rapor vs Alarm Farkı

| Özellik | Periyodik Rapor | Alarm |
|---------|-----------------|-------|
| Tetikleyici | Zaman (60s) | Olay/koşul |
| Medya | snapshot.jpg (üzerine yazar) | timestamp.jpg (benzersiz) |
| API Endpoint | `/ai/input` | `/ai/alarm` |
| Cooldown | — | 5–30s |

---

## 10. VERİ AKIŞI ÖRNEKLERİ

### 10.1 Periyodik Rapor Akışı (Kapı Kamerası)

```
1. track_video() ana thread
   ├── report.check_reports()
   │   └── 60 saniye geçti mi? EVET
   │
   ├── modules.get_data()
   │   └── {"Total Open Seconds": 240}
   │
   ├── minio.upload_report(frame, "ingiltere-kamera-1")
   │   └── "kafkas/01-06-2026/ingiltere-kamera-1/snapshot.jpg"
   │
   └── report.send_report("kapi_raporu", data, media_path)
       └── Kuyruğa ekler
           ↓
2. _sender_thread (arka plan)
   └── POST https://app.youreye.co.uk/api/ai/input
```

### 10.2 Alarm Akışı (Kapıda Kişi Yok)

```
1. door_module.update()
   ├── ROI piksel std_dev hesapla → AÇIK
   ├── Kişi polygon'unda yok
   ├── 30 saniye geçti
   └── _open_duration_alert = True

2. modules.get_data()
   └── {"open_duration_alert": True}

3. Alarm kontrol
   ├── trigger_on: "open_duration_alert" UYUYOR
   ├── Cooldown (25s) geçti → EVET
   ├── minio.upload_alert() → "kapida_kisi_yok_10-30-45.jpg"
   └── report.send_alarm("kapida_kisi_yok", ...)
       ↓
4. _sender_thread
   └── POST https://app.youreye.co.uk/api/ai/alarm
```

---

## 11. DURUM KALICILIĞI

### 11.1 Durum Dosyaları — `state/` Dizini

| Dosya | Modül | İçerik |
|-------|-------|--------|
| `door_open_closed_*.json` | DoorOpenClosed | daily_opened, total_open_sec |
| `calisma_suresi_*.json` | CalismaSuresi | alan toplamları, günlük toplam |
| `forklift_*.json` | ForkliftSuresi | total_working_seconds |
| `health_ingiltere-kamera-*.json` | Sistem | connected (bool), timestamp |
| `lens_detector_state.json` | LensDetector | baseline, bulanıklık değerleri |

**Kaydetme Mantığı:**
- Her 30 saniyede bir
- Kapanışta (graceful)
- Tarih değişirse → sayaçları sıfırla
- Thread-güvenli: `.tmp` yazar, ardından atomik yeniden adlandırma

### 11.2 Kuyruk Günlükleri

- Dosya: `queue_log_{minio_folder}.json`
- Format: JSONL (satır başına bir JSON)
- Ağ kesintisinde API kuyruğunu korur
- Başlangıçta yüklenir
- Maks. 500 öğe

---

## 12. HATA YÖNETİMİ VE DAYANIKLILIK

| Hata Türü | Tespit | Kurtarma |
|-----------|--------|----------|
| CUDA/GPU hatası | Detection thread | `_reload_model()` çağrısı |
| Akış kesilmesi | Cap thread | Üstel geri çekilme ile yeniden bağlantı (maks. 60s) + alarm |
| MinIO hatası | `upload_image()` | `None` döner, program devam eder |
| API hatası | Gönderici thread | Yerel kuyruğa ekler, 5s sonra yeniden dener |
| Process çökmesi | Süpervizör thread | 15s sonra yeniden başlatır (maks. 10 deneme) |
| Graceful kapatma | SIGINT/SIGTERM | Thread'ler durdurulur, durum kaydedilir, 5s sonra zorla çıkar |

---

## 13. SINIF HİYERARŞİSİ

```
BaseModule (soyut arayüz)
  ├── CalismaSuresiModule
  ├── DoorOpenClosedModule
  ├── DoorReportModule
  ├── ForkliftSuresiModule
  ├── HardwareMonitorModule
  ├── RestrictedZoneModule
  ├── FaceBlurringModule
  ├── LensDetectorModule
  └── WarehousePersonCounterModule

ModuleRunner
  └── tüm modülleri yönetir

ReportManager
  ├── Periyodik raporlar
  ├── Alarmlar
  ├── _sender_thread (arka plan HTTP gönderici)
  └── Kuyruk kalıcılığı

MinIOManager
  ├── upload_report()
  └── upload_alert()

MultiThreadingTracker
  ├── Cap thread
  ├── ObjectDetection (YOLO)
  └── OCSoft / StrongSORT

ObjectDetection
  ├── process_image() (thread döngüsü)
  └── detect_batch() (çıkarım)

MultiObjectTracking
  ├── ocsort()      → OCSort
  ├── strongsort()  → StrongSORT
  └── bytetrack()   → BYTETracker (devre dışı)
```

---

## 14. PERFORMANS KARAKTERİSTİKLERİ

### 14.1 Eşzamanlılık Modeli

| Katman | Yöntem | Birim başına |
|--------|--------|-------------|
| Kameralar arası | `multiprocessing.Pool` | 1 process/kamera |
| Kamera içi (Cap + Tespit) | `threading.Thread` | 2 thread/kamera |
| API gönderme | Arka plan thread | 1 thread/kamera |

### 14.2 Bellek Kullanımı

| Bileşen | Bellek |
|---------|--------|
| Kare tamponları (1080p × 4) | ~12 MB/kamera |
| YOLO8n modeli | ~6 MB |
| YOLO26m-pose modeli | ~200 MB |
| Özel forklift modeli | ~10–50 MB |
| Kamera başına toplam | ~200–300 MB RAM |

### 14.3 Gecikme

| Aşama | Süre |
|-------|------|
| Kare → Tespit | 100–200 ms |
| Tespit → Takip | <10 ms |
| Takip → Modül | <50 ms |
| Modül → API | Asenkron (arka plan) |
| Uçtan uca | ~200–300 ms/kare |

---

## 15. BAŞLATMA NOKTALARI

### Geliştirme Ortamı

```bash
python kafkas/main.py
```

- Yapılandırma yükler
- N kamera için N süreç başlatır
- Ctrl+C ile düzgün kapatır

### Üretim Ortamı

```bash
python kafkas/run.py
```

- `main.py`'yi alt süreç olarak başlatır
- Stdout/stderr'i günlük dosyasına yönlendirir
- Çökme durumunda otomatik yeniden başlatır
- Eski günlükleri temizler (son 10 tutar)

### Test Araçları

| Script | Amaç |
|--------|------|
| `api_test.py` | API bağlantısını test et |
| `minio_test.py` | MinIO bağlantısını test et |
| `is_software_alive.py` | Sağlık durumu kontrolü |
| `roi_ciz.py` | ROI çizim aracı |
| `rescale_rois.py` | ROI koordinat ölçeklendirici |
| `make_reference.py` | Referans görüntü oluştur |

---

## 16. TEKNOLOJİ YIĞINI

| Katman | Teknoloji |
|--------|-----------|
| **Dil** | Python 3.8+ |
| **Bilgisayarlı Görü** | OpenCV (cv2), Ultralytics YOLO |
| **Derin Öğrenme** | PyTorch, CUDA |
| **Nesne Takibi** | OCSoft (varsayılan), StrongSORT, ByteTrack |
| **Depolama** | MinIO (S3 uyumlu) |
| **API** | requests (HTTP) |
| **Eşzamanlılık** | multiprocessing, threading, Queue |
| **Yapılandırma** | JSON dosyaları |
| **Kalıcılık** | JSON durum dosyaları, MinIO |
| **İzleme** | Özel sağlık kontrolleri, Healthcheck.io |

---

## 17. BİLİNEN SORUNLAR VE NOTLAR

**`BUGLAR_VE_COZUMLER.md` dosyasından:**
1. Uzun süreli çalışmada CUDA bellek sızıntıları
2. GStreamer boru hattı stabilitesi kameraya göre değişiyor
3. Yüz bulanıklaştırma FPS'i etkiliyor (~10 FPS kayıp)
4. Kapı tespiti ışık değişimlerine duyarlı

**Koddan tespit edilen:**
1. ByteTrack devre dışı (yorum satırı)
2. Yinelenen modüller: `warehousePersonCounter.py` vs `warehouse_person_counter.py`
3. Yinelenen modüller: `restrictedZoneDetection.py` vs `restricted_zone_module.py`
4. Yüz bulanıklaştırma modeli (`yolov8l_100e.pt`) büyük (~200 MB)
5. İzleyici seçimi şu anda hardcoded (OCSoft varsayılan)

---

## 18. PROJE İSTATİSTİKLERİ

| Metrik | Değer |
|--------|-------|
| Toplam Python dosyası | 30+ |
| Ana kod satır sayısı | ~2.048 satır (7 ana dosya) |
| Uygulanan modül | 9 |
| Yapılandırılmış kamera | 6 (3 aktif izleme) |
| Aktif izleyici algoritma | 2 (1 devre dışı) |
| Markdown belge dosyası | 7 |
| JSON yapılandırma dosyası | 13 |
| Durum dosyaları | Dinamik (kamera/modül başına) |

---

## 19. MİMARİ DEĞERLENDİRME

### Güçlü Yönler

- Modüler tasarım (BaseModule arayüzü ile kolayca genişletilebilir)
- Çok kameralı paralelleştirme (multiprocessing)
- Çevrimdışı kuyruk dayanıklılığı
- Kapsamlı hata yönetimi
- Durum kalıcılığı (günlük sıfırlama ile)
- Esnek alarm sistemi (soğuma, günde bir kez modu)
- MinIO ile API arasında gevşek bağlantı

### Potansiyel İyileştirme Alanları

- Dosya tabanlı durum yerine hafif veritabanı (SQLite)
- Prometheus gibi metrik toplama entegrasyonu
- Yinelenen modül dosyalarının birleştirilmesi
- ByteTrack'in test edilmesi ve karşılaştırılması
- Yeniden başlatma gerektirmeden yapılandırma hot-reload
- Birim testlerinin eklenmesi (mevcut değil)
