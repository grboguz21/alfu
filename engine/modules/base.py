class BaseModule:
    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        """Called on every frame."""
        raise NotImplementedError

    def get_data(self) -> dict:
        """Called when a periodic report is triggered."""
        raise NotImplementedError

    def draw(self, frame):
        """Modules that draw on the frame override this."""
        return frame

    def reset(self):
        """Used to reset statistics manually or automatically."""
        pass

    def shutdown(self):
        """Modules that write state to disk on shutdown override this."""
        pass
