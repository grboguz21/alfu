from .face_blurring            import FaceBlurringModule
from .lens_detector            import LensDetectorModule
from .market_client_counter    import MarketClientCounterModule
from .cashier_working_hours    import KasiyerSuresiModule
from .bolge_vakit_analizi      import BolgeVakitAnaliziModule
from .office_hours             import WorkHoursModule

def _scale_pt(pt, ratio: float):
    return [int(pt[0] * ratio), int(pt[1] * ratio)]

def _scale_pts(pts, ratio: float):
    return [_scale_pt(p, ratio) for p in pts]

def _scale_module_coords(params: dict, ratio: float) -> dict:
    """polygon/roi/filter_line/check_points koordinatlarını ratio ile ölçekler."""
    params = dict(params)
    if "polygon" in params:
        params["polygon"] = _scale_pts(params["polygon"], ratio)
    if "polygon_in" in params:
        params["polygon_in"] = _scale_pts(params["polygon_in"], ratio)
    if "polygon_out" in params:
        params["polygon_out"] = _scale_pts(params["polygon_out"], ratio)
    if "zone" in params and params["zone"]:
        params["zone"] = _scale_pts(params["zone"], ratio)
    if "filter_line" in params and params["filter_line"]:
        params["filter_line"] = _scale_pts(params["filter_line"], ratio)
    if "check_points" in params and params["check_points"]:
        params["check_points"] = {
            k: _scale_pt(v, ratio) for k, v in params["check_points"].items()
        }
    if "stations" in params:
        scaled = []
        for st in params["stations"]:
            st = dict(st)
            if "roi" in st:
                st["roi"] = _scale_pts(st["roi"], ratio)
            scaled.append(st)
        params["stations"] = scaled
    return params

MODULE_REGISTRY = {
    "face_blurring":           FaceBlurringModule,
    "lens_detector":           LensDetectorModule,
    "market_client_counter":   MarketClientCounterModule,
    "kasiyer_suresi":          KasiyerSuresiModule,
    "bolge_vakit_analizi":     BolgeVakitAnaliziModule,
    "work_hours":              WorkHoursModule,
}


class ModuleRunner:
    """Loads modules from config list, runs them on each frame, and collects data."""

    def __init__(self, modules_cfg: list, frame_resize: float = 1.0):
        self.modules = []
        # Sınıf düzeyinde (self.) tanımladık ki get_data içerisinden erişebilelim
        self.disabled_modules = []

        scale = frame_resize

        for cfg in modules_cfg:
            module_type = cfg.get("type")

            # Modül listedeyse veya registry'de tanımlı değilse hata vermeden geç
            if module_type in self.disabled_modules or module_type not in MODULE_REGISTRY:
                print(f"⚠️ Modul gecildi (Yazilimda devre disi): {module_type}")
                continue

            cls = MODULE_REGISTRY.get(module_type)
            assert cls is not None, f"Unknown module type: {module_type}"

            params = {k: v for k, v in cfg.items() if k != "type"}
            if scale != 1.0:
                params = _scale_module_coords(params, scale)
            self.modules.append(cls(**params))
            print(f"✅ Module loaded: {module_type} → {cfg.get('name', '')}")

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        for module in self.modules:
            module.update(bboxes, class_ids, scores, object_ids, frame, class_names)

    def get_data(self) -> dict:
        data = {}
        for module in self.modules:
            data.update(module.get_data())

        return data

    def draw(self, frame):
        for module in self.modules:
            frame = module.draw(frame)
        return frame

    def update_all_modules(self, func):
        """Applies a function to all modules (e.g. reset)."""
        for module in self.modules:
            func(module)

    def shutdown(self):
        for module in self.modules:
            module.shutdown()
        print("✅ Module states saved.")

    def reset(self):
        for module in self.modules:
            module.reset()
        print("✅ All modules reset.")