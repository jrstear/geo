from app.plugins import PluginBase, MountPoint

from .api_views import GenerateGCPView, DownloadGCPView


class Plugin(PluginBase):
    def include_js_files(self):
        return ["load_buttons.js"]

    def build_jsx_components(self):
        return ["TaskView.jsx"]

    def api_mount_points(self):
        return [
            MountPoint("task/(?P<pk>[^/.]+)/generate", GenerateGCPView.as_view()),
            MountPoint("task/(?P<pk>[^/.]+)/download/(?P<filename>[^/]+)", DownloadGCPView.as_view()),
        ]
