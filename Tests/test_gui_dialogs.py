import pytest
from src.annotator_dashboard.dialogs import SettingsDialog

def test_settings_dialog_init(qapp):
    dialog = SettingsDialog(parent=None)
    assert dialog.windowTitle() == "Settings"
    assert dialog.chk_rotate.isChecked() is True
    assert dialog.chk_reproject.isChecked() is False
    assert dialog.chk_show_confidence.isChecked() is True
