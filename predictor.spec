# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['predictor.py'],
             pathex=['/Users/yunzhang/Desktop/Projects/PaO2Predictor'],
             binaries=[],
             datas=[('saved_models/regressor', 'saved_models/regressor')],
             hiddenimports=['sklearn.neural_network'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='PaO2Predictor.exe',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )


app = BUNDLE(exe,
         name='PaO2Predictor.app',
         icon=None,
         bundle_identifier=None)
