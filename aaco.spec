# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for AACO CLI executable.

Build with:
    pyinstaller aaco.spec

Output:
    dist/aaco.exe (Windows)
    dist/aaco (Linux/macOS)
"""

import sys
from pathlib import Path


block_cipher = None

# Get the aaco package path
aaco_path = Path('.').resolve()

a = Analysis(
    ['aaco/__main__.py'],
    pathex=[str(aaco_path)],
    binaries=[],
    datas=[
        # Include any data files needed by the package
        ('aaco/templates', 'aaco/templates'),
    ],
    hiddenimports=[
        # Core dependencies
        'click',
        'rich',
        'rich.console',
        'rich.table',
        'rich.panel',
        'rich.progress',
        'numpy',
        'pandas',
        'scipy',
        'scipy.cluster',
        'scipy.cluster.hierarchy',
        'scipy.stats',
        'matplotlib',
        'matplotlib.pyplot',
        'seaborn',
        'jinja2',
        'yaml',
        'psutil',
        # AACO modules
        'aaco',
        'aaco.cli',
        'aaco.core',
        'aaco.core.session',
        'aaco.core.schema',
        'aaco.runner',
        'aaco.runner.ort_runner',
        'aaco.collectors',
        'aaco.collectors.sys_sampler',
        'aaco.collectors.rocm_smi_sampler',
        'aaco.collectors.clocks',
        'aaco.analytics',
        'aaco.analytics.metrics',
        'aaco.analytics.classify',
        'aaco.analytics.diff',
        'aaco.profiler',
        'aaco.report',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude optional heavy dependencies
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'streamlit',
        'plotly',
        'onnx',
        'onnxruntime',
        'torch',
        'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='aaco',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if desired: 'assets/aaco.ico'
)
