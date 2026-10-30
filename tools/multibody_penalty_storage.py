"""Approved external output/cache storage; retain every original hardware gate."""
import os
from pathlib import Path
import shutil

from guard_cylinder_run import sample, rejection
from multibody_penalty_config import EXTERNAL, ROOT


class ExternalStorage:
    def prepare(self, output):
        self.mount = Path('/media/user/external')
        if not self.mount.is_mount() or not output.is_relative_to(EXTERNAL):
            raise ValueError('authorised mounted external experiment directory required')
        self.device = self.mount.stat().st_dev
        if self.device == ROOT.stat().st_dev:
            raise ValueError('external output must not fall back to the working filesystem')
        EXTERNAL.mkdir(exist_ok=True)
        self.initial_system_free = shutil.disk_usage(ROOT).free / 1024**3
        self.runtime = EXTERNAL / 'runtime' / output.name
        self.runtime.mkdir(parents=True, exist_ok=False)
        self.environment = {'PYTHONDONTWRITEBYTECODE': '1'}
        for variable, name in (('TMPDIR','tmp'), ('XDG_CACHE_HOME','xdg'),
                               ('MPLCONFIGDIR','matplotlib'), ('CUDA_CACHE_PATH','cuda'),
                               ('JAX_COMPILATION_CACHE_DIR','jax')):
            path = self.runtime / name
            path.mkdir()
            self.environment[variable] = str(path)
        self.environment['TMP'] = self.environment['TEMP'] = self.environment['TMPDIR']
        return self.environment

    def sample(self, path):
        if not self.mount.is_mount() or self.mount.stat().st_dev != self.device:
            raise RuntimeError('external volume disconnected or replaced')
        values = sample(path)
        values['system_disk_free_gib'] = shutil.disk_usage(ROOT).free / 1024**3
        values['system_disk_loss_gib'] = self.initial_system_free-values['system_disk_free_gib']
        return values

    def rejection(self, values, *, startup, low_disk_exception):
        reason = rejection(values, startup=startup, low_disk_exception=low_disk_exception)
        if reason:
            return reason
        if values['system_disk_loss_gib'] > .125:
            return 'system volume lost over 128 MiB since launch; external storage scope only'
        total = sum(p.stat().st_size for p in self.runtime.rglob('*') if p.is_file())
        if total > 256*1024**2:
            return 'external runtime/cache budget exceeds 256 MiB'
        return None

    def metadata(self):
        return dict(external_root=str(EXTERNAL), mounted_device=self.device,
                    external_minimum_free_gib=30, system_initial_free_gib=self.initial_system_free,
                    system_maximum_loss_mib=128, runtime_maximum_mib=256,
                    environment=self.environment,
                    external_drive_temperature='SMART unavailable to current user; not measured')
