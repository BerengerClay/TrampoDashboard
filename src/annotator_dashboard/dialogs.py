import os

from PyQt6.QtWidgets import (
    QDialog,
    QCheckBox,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QListView,
    QTreeView,
    QAbstractItemView,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
)
from icons import configure_button
from PyQt6.QtCore import Qt


def select_multiple_directories(
    parent=None, caption="Select Directories", directory=""
):
    """Opens a non-native file dialog allowing multiple directories to be selected."""
    dialog = QFileDialog(parent, caption, directory)
    dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
    dialog.setFileMode(QFileDialog.FileMode.Directory)

    # Enable multiple/extended selection in the internal view widget
    for view in dialog.findChildren((QListView, QTreeView)):
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.selectedFiles()
    return []


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_win = parent
        self.setWindowTitle("Settings")
        self.resize(420, 360)

        if parent:
            self.original_kp_radius = parent.keypoint_radius
            self.original_rotate = parent.auto_rotate_enabled
            self.original_reproject = parent.show_3d_reprojection
            self.original_show_confidence = getattr(
                parent, "vitpose_show_confidence", True
            )
            self.original_kp_size_3d = getattr(parent, "keypoint_size_3d", 50)
            self.original_show_gt = getattr(parent, "show_gt_overlay", False)
            self.original_kalman_enabled = getattr(parent, "kalman_enabled", True)
            self.original_kalman_q = getattr(parent, "kalman_q", 0.0001)
            self.original_kalman_r = getattr(parent, "kalman_r", 0.002)
            self.original_use_kalman_trc = getattr(parent, "use_kalman_trc", False)
            self.original_kalman_overlay = getattr(parent, "show_kalman_overlay", False)
        else:
            self.original_kp_radius = 3
            self.original_rotate = True
            self.original_reproject = False
            self.original_show_confidence = True
            self.original_kp_size_3d = 50
            self.original_show_gt = False
            self.original_kalman_enabled = True
            self.original_kalman_q = 0.01
            self.original_kalman_r = 0.1
            self.original_use_kalman_trc = False
            self.original_kalman_overlay = False

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Auto-rotation checkbox
        self.chk_rotate = QCheckBox(
            "Keep feet at bottom and head at top (auto-rotation)"
        )
        self.chk_rotate.setChecked(parent.auto_rotate_enabled if parent else True)
        if parent:
            self.chk_rotate.toggled.connect(self.on_rotate_toggled)
        layout.addWidget(self.chk_rotate)

        # Reprojection checkbox
        self.chk_reproject = QCheckBox("Show 3D reprojection overlay (Red)")
        self.chk_reproject.setChecked(parent.show_3d_reprojection if parent else False)
        if parent:
            self.chk_reproject.toggled.connect(self.on_reproject_toggled)
        layout.addWidget(self.chk_reproject)

        # Kalman Reprojection Overlay checkbox
        self.chk_kalman_overlay = QCheckBox("Show Kalman 3D reprojection overlay (Purple)")
        self.chk_kalman_overlay.setChecked(getattr(parent, "show_kalman_overlay", False) if parent else False)
        if parent:
            self.chk_kalman_overlay.toggled.connect(self.on_kalman_overlay_toggled)
        layout.addWidget(self.chk_kalman_overlay)

        # Show confidence checkbox
        self.chk_show_confidence = QCheckBox("Show ViTPose confidence (opacity based on confidence)")
        self.chk_show_confidence.setChecked(parent.vitpose_show_confidence if parent else True)
        if parent:
            self.chk_show_confidence.toggled.connect(self.on_show_confidence_toggled)
        layout.addWidget(self.chk_show_confidence)

        # Show Ground Truth (GT) overlay checkbox
        self.chk_show_gt = QCheckBox("Show Ground Truth (GT) overlay")
        self.chk_show_gt.setChecked(parent.show_gt_overlay if parent else False)
        if parent:
            self.chk_show_gt.toggled.connect(self.on_gt_toggled)
        layout.addWidget(self.chk_show_gt)

        # Keypoint size slider layout
        kp_size_layout = QHBoxLayout()
        kp_size_lbl = QLabel("Keypoint Size:")
        kp_size_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")

        self.slider_kp_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_kp_size.setRange(1, 10)
        self.slider_kp_size.setValue(self.original_kp_radius)
        if parent:
            self.slider_kp_size.valueChanged.connect(parent.update_keypoint_sizes)

        self.slider_kp_size.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38bdf8;
                border: 1px solid #0284c7;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        kp_size_layout.addWidget(kp_size_lbl)
        kp_size_layout.addWidget(self.slider_kp_size)
        layout.addLayout(kp_size_layout)

        # 3D Keypoint size slider layout
        kp_size_3d_layout = QHBoxLayout()
        kp_size_3d_lbl = QLabel("3D Keypoint Size:")
        kp_size_3d_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")

        self.slider_kp_size_3d = QSlider(Qt.Orientation.Horizontal)
        self.slider_kp_size_3d.setRange(10, 150)
        self.slider_kp_size_3d.setValue(self.original_kp_size_3d)
        if parent:
            self.slider_kp_size_3d.valueChanged.connect(self.on_kp_size_3d_changed)

        self.slider_kp_size_3d.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #3b82f6;
                border: 1px solid #1d4ed8;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        kp_size_3d_layout.addWidget(kp_size_3d_lbl)
        kp_size_3d_layout.addWidget(self.slider_kp_size_3d)
        layout.addLayout(kp_size_3d_layout)




        # Help & Controls Box
        help_group = QGroupBox("Keyboard Shortcuts & Controls")
        help_group.setStyleSheet("""
            QGroupBox {
                color: #38bdf8;
                font-weight: bold;
                border: 1px solid #334155;
                margin-top: 10px;
                padding-top: 15px;
                border-radius: 6px;
                font-size: 12px;
            }
        """)
        help_layout = QVBoxLayout(help_group)

        help_text = (
            "<b>Mouse Interaction:</b><br>"
            "• Double-click view: Maximize/Minimize the camera view<br>"
            "• Drag (Left Click): Pan camera canvas<br>"
            "• Mouse Wheel: Zoom in/out of camera canvas<br><br>"
            "<b>Global Shortcuts:</b><br>"
            "  - <b>Left / Right Arrow</b>: Frame Navigation<br>"
            "  - <b>Escape</b>: Reset to 8-view Grid Mode"
        )
        lbl_help = QLabel(help_text)
        lbl_help.setWordWrap(True)
        lbl_help.setStyleSheet("color: #94a3b8; font-size: 11px; line-height: 1.4;")
        help_layout.addWidget(lbl_help)
        layout.addWidget(help_group)

        layout.addSpacing(10)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QCheckBox {
                color: #f8fafc;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #38bdf8;
                border-color: #0284c7;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

    def on_rotate_toggled(self, checked):
        if self.parent_win:
            self.parent_win.auto_rotate_enabled = checked
            # Refresh camera orientations immediately
            for cam in self.parent_win.camera_widgets:
                if cam.view_mode == "bbox":
                    cam.apply_bbox_view()

    def on_reproject_toggled(self, checked):
        if self.parent_win:
            self.parent_win.show_3d_reprojection = checked
            self.parent_win.show_current_frame(preserve_view=True)

    def on_kalman_overlay_toggled(self, checked):
        if self.parent_win:
            self.parent_win.show_kalman_overlay = checked
            self.parent_win.show_current_frame(preserve_view=True)

    def on_show_confidence_toggled(self, checked):
        if self.parent_win:
            self.parent_win.vitpose_show_confidence = checked
            self.parent_win.show_current_frame(preserve_view=True)

    def on_gt_toggled(self, checked):
        if self.parent_win:
            self.parent_win.show_gt_overlay = checked
            self.parent_win.show_current_frame(preserve_view=True)
            self.parent_win.update_3d_view()

    def on_kp_size_3d_changed(self, value):
        if self.parent_win:
            self.parent_win.keypoint_size_3d = value
            # Update inline visualizer if present
            if hasattr(self.parent_win, "visualizer_3d_inline") and self.parent_win.visualizer_3d_inline:
                self.parent_win.visualizer_3d_inline.keypoint_size = value
            # Update popout visualizer if present
            if hasattr(self.parent_win, "visualizer_3d_window") and self.parent_win.visualizer_3d_window:
                self.parent_win.visualizer_3d_window.widget_3d.keypoint_size = value
                self.parent_win.visualizer_3d_window.update_visualization()
            self.parent_win.update_3d_view()

    def q_to_slider(self, q):
        import numpy as np
        return int(np.clip(1 + (np.log10(max(1e-5, q)) - (-5)) * 99 / 4.0, 1, 100))

    def slider_to_q(self, val):
        return float(10 ** (-5.0 + (val - 1) * 4.0 / 99.0))

    def r_to_slider(self, r):
        import numpy as np
        return int(np.clip(1 + (np.log10(max(1e-4, r)) - (-4)) * 99 / 4.0, 1, 100))

    def slider_to_r(self, val):
        return float(10 ** (-4.0 + (val - 1) * 4.0 / 99.0))

    def on_trc_mode_changed(self, index):
        use_kalman = (index == 1)
        if self.parent_win and hasattr(self.parent_win, "on_kalman_mode_changed"):
            self.parent_win.on_kalman_mode_changed(use_kalman)

    def on_run_kalman_clicked(self):
        q = self.slider_to_q(self.slider_q.value())
        r = self.slider_to_r(self.slider_r.value())
        if self.parent_win and hasattr(self.parent_win, "run_and_save_kalman_trc"):
            self.kalman_status_lbl.setText("Génération du TRC Kalman en cours...")
            self.kalman_status_lbl.repaint()
            success = self.parent_win.run_and_save_kalman_trc(q, r)
            if success:
                self.combo_trc_mode.blockSignals(True)
                self.combo_trc_mode.setCurrentIndex(1)
                self.combo_trc_mode.blockSignals(False)
                self.kalman_status_lbl.setText("Généré avec succès !")
            else:
                self.kalman_status_lbl.setText("Erreur lors de la génération.")

    def on_kalman_slider_changed(self, _value):
        q = self.slider_to_q(self.slider_q.value())
        r = self.slider_to_r(self.slider_r.value())
        self.q_val_lbl.setText(f"{q:.5f}")
        self.r_val_lbl.setText(f"{r:.4f}")

    def accept(self):
        if self.parent_win:
            self.parent_win.save_local_settings()
        super().accept()

    def reject(self):
        if self.parent_win:
            self.parent_win.update_keypoint_sizes(self.original_kp_radius)
            self.parent_win.auto_rotate_enabled = self.original_rotate
            self.parent_win.show_3d_reprojection = self.original_reproject
            self.parent_win.show_kalman_overlay = self.original_kalman_overlay
            self.parent_win.vitpose_show_confidence = self.original_show_confidence
            self.parent_win.show_gt_overlay = self.original_show_gt
            
            # Revert 3D keypoint size
            self.parent_win.keypoint_size_3d = self.original_kp_size_3d
            if hasattr(self.parent_win, "visualizer_3d_inline") and self.parent_win.visualizer_3d_inline:
                self.parent_win.visualizer_3d_inline.keypoint_size = self.original_kp_size_3d
            if hasattr(self.parent_win, "visualizer_3d_window") and self.parent_win.visualizer_3d_window:
                self.parent_win.visualizer_3d_window.widget_3d.keypoint_size = self.original_kp_size_3d
                self.parent_win.visualizer_3d_window.update_visualization()

            # Revert 3D Kalman mode
            if hasattr(self.parent_win, "on_kalman_mode_changed"):
                self.parent_win.on_kalman_mode_changed(self.original_use_kalman_trc)
                
            # Re-apply orientations
            for cam in self.parent_win.camera_widgets:
                if cam.view_mode == "bbox":
                    cam.apply_bbox_view()
            self.parent_win.show_current_frame(preserve_view=True)
        super().reject()


class SelectCameraFoldersDialog(QDialog):
    def __init__(
        self, camera_keys, initial_parent="", prefilled_dirs=None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Folders")
        self.resize(650, 450)
        self.camera_keys = camera_keys
        self.camera_dirs = {}
        self.initial_parent = initial_parent

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Style sheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 4px 10px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

        main_layout.addWidget(QLabel("<b>Individual Camera Folders:</b>"))

        # 8 Camera folders rows
        self.cam_inputs = {}
        for key in camera_keys:
            row_layout = QHBoxLayout()
            cam_lbl = QLabel(f"{key}:")
            cam_lbl.setMinimumWidth(120)

            initial_val = prefilled_dirs.get(key, "") if prefilled_dirs else ""
            cam_txt = QLineEdit(initial_val)
            btn_cam_browse = QPushButton("Browse...")

            # Use default capture in lambda
            btn_cam_browse.clicked.connect(
                lambda checked=False, k=key: self.browse_camera(k)
            )

            row_layout.addWidget(cam_lbl)
            row_layout.addWidget(cam_txt, stretch=1)
            row_layout.addWidget(btn_cam_browse)
            main_layout.addLayout(row_layout)
            self.cam_inputs[key] = cam_txt

        main_layout.addStretch()

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

    def browse_camera(self, key):
        initial = self.cam_inputs[key].text()
        if not initial:
            initial = self.initial_parent
        dir_path = QFileDialog.getExistingDirectory(
            self, f"Select Folder for {key}", initial
        )
        if dir_path:
            self.cam_inputs[key].setText(dir_path)

    def validate_and_accept(self):
        # Retrieve and validate directories
        dirs = {}
        for key, input_widget in self.cam_inputs.items():
            path = input_widget.text().strip()
            if not path:
                QMessageBox.warning(
                    self, "Missing Folder", f"Please select a directory for {key}."
                )
                return
            if not os.path.isdir(path):
                QMessageBox.warning(
                    self,
                    "Invalid Folder",
                    f"The directory for {key} does not exist:\n{path}",
                )
                return
            # Check if directory is empty or has no images
            files = os.listdir(path)
            has_images = any(
                f.lower().endswith((".png", ".jpg", ".jpeg")) for f in files
            )
            if not has_images:
                QMessageBox.warning(
                    self,
                    "No Images",
                    f"The directory for {key} does not contain any images (.png, .jpg, .jpeg):\n{path}",
                )
                return
            dirs[key] = path

        self.camera_dirs = dirs
        self.accept()

    def get_camera_dirs(self):
        return self.camera_dirs


class PreprocessOptionsDialog(QDialog):
    def __init__(self, total_frames, current_frame_idx=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pre-processing Options")
        self.resize(400, 320)
        self.total_frames = total_frames
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Style sheet
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-size: 12px;
            }
            QSpinBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QCheckBox, QRadioButton {
                color: #f8fafc;
                font-size: 12px;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 16px;
                height: 16px;
                background-color: #1e293b;
                border: 1px solid #475569;
            }
            QCheckBox::indicator {
                border-radius: 4px;
            }
            QRadioButton::indicator {
                border-radius: 8px;
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #38bdf8;
                border-color: #0284c7;
            }
            QPushButton {
                background-color: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

        layout.addWidget(QLabel("<b>Pre-processing Configuration:</b>"))

        # Radio buttons to run preprocessing
        self.rb_yolo_vitpose = QRadioButton("Run pre-processing (YOLO + ViTPose)")
        self.rb_yolo = QRadioButton("Run pre-processing (YOLO only)")
        self.rb_none = QRadioButton("No pre-processing (Step / Interpolation only)")
        self.rb_yolo_vitpose.setChecked(True)

        self.btn_group = QButtonGroup(self)
        self.btn_group.addButton(self.rb_yolo_vitpose)
        self.btn_group.addButton(self.rb_yolo)
        self.btn_group.addButton(self.rb_none)

        layout.addWidget(self.rb_yolo_vitpose)
        layout.addWidget(self.rb_yolo)
        layout.addWidget(self.rb_none)

        # Start frame
        start_layout = QHBoxLayout()
        start_lbl = QLabel("Starting frame index:")
        start_lbl.setMinimumWidth(180)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, total_frames)
        self.spin_start.setValue(current_frame_idx + 1)
        start_layout.addWidget(start_lbl)
        start_layout.addWidget(self.spin_start)
        layout.addLayout(start_layout)

        # Frame step
        step_layout = QHBoxLayout()
        step_lbl = QLabel("Frame step:")
        step_lbl.setMinimumWidth(180)
        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)
        self.spin_step.setValue(8)
        step_layout.addWidget(step_lbl)
        step_layout.addWidget(self.spin_step)
        layout.addLayout(step_layout)

        # Explanation box
        explanation = QLabel(
            "<i>Note: Intermediate frames (skipped by the step) will "
            "be automatically calculated by linear interpolation between the processed frames.</i>"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(explanation)

        layout.addStretch()

        # Dialog buttons
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        mode = "none"
        if self.rb_yolo_vitpose.isChecked():
            mode = "yolo_vitpose"
        elif self.rb_yolo.isChecked():
            mode = "yolo"

        return {
            "run_preprocess": mode != "none",
            "preprocess_mode": mode,
            "start_frame_idx": self.spin_start.value() - 1,
            "frame_step": self.spin_step.value(),
        }


class KalmanSettingsDialog(QDialog):
    """Dedicated dialog for configuring Kalman RTS filter parameters (Q, R) and generating triangulated_kalman.trc."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTS Kalman Filter Settings")
        self.resize(460, 340)
        self.parent_win = parent

        if parent:
            self.original_kalman_q = getattr(parent, "kalman_q", 0.01)
            self.original_kalman_r = getattr(parent, "kalman_r", 0.1)
            self.original_use_kalman_trc = getattr(parent, "use_kalman_trc", False)
        else:
            self.original_kalman_q = 0.01
            self.original_kalman_r = 0.1
            self.original_use_kalman_trc = False

        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QLabel {
                color: #f8fafc;
                font-size: 12px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # Title / Description Box
        desc_lbl = QLabel("Configuration of the Rauch-Tung-Striebel (RTS) Kalman Filter and smoothed 3D trajectory generation.")
        desc_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        # 3D TRC Mode Selector
        mode_layout = QHBoxLayout()
        mode_lbl = QLabel("3D Trajectory Source:")
        mode_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 12px;")
        
        self.combo_trc_mode = QComboBox()
        self.combo_trc_mode.addItems(["Raw (triangulated.trc)", "Kalman Filtered (triangulated_kalman.trc)"])
        self.combo_trc_mode.setCurrentIndex(1 if (parent and getattr(parent, "use_kalman_trc", False)) else 0)
        self.combo_trc_mode.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
        """)
        self.combo_trc_mode.currentIndexChanged.connect(self.on_trc_mode_changed)
        mode_layout.addWidget(mode_lbl)
        mode_layout.addWidget(self.combo_trc_mode)
        layout.addLayout(mode_layout)

        # Q Slider (Process Noise)
        q_layout = QHBoxLayout()
        q_lbl = QLabel("Process Noise (Q):")
        q_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 11px;")
        self.q_val_lbl = QLabel(f"{self.original_kalman_q:.5f}")
        self.q_val_lbl.setStyleSheet("color: #c084fc; font-weight: bold; font-size: 11px; min-width: 65px;")

        self.slider_q = QSlider(Qt.Orientation.Horizontal)
        self.slider_q.setRange(1, 100)
        self.slider_q.setValue(self.q_to_slider(self.original_kalman_q))
        self.slider_q.valueChanged.connect(self.on_kalman_slider_changed)
        self.slider_q.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #a855f7;
                border: 1px solid #7e22ce;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        q_layout.addWidget(q_lbl)
        q_layout.addWidget(self.slider_q)
        q_layout.addWidget(self.q_val_lbl)
        layout.addLayout(q_layout)

        # R Slider (Measurement Noise)
        r_layout = QHBoxLayout()
        r_lbl = QLabel("Measurement Noise (R):")
        r_lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 11px;")
        self.r_val_lbl = QLabel(f"{self.original_kalman_r:.4f}")
        self.r_val_lbl.setStyleSheet("color: #c084fc; font-weight: bold; font-size: 11px; min-width: 65px;")

        self.slider_r = QSlider(Qt.Orientation.Horizontal)
        self.slider_r.setRange(1, 100)
        self.slider_r.setValue(self.r_to_slider(self.original_kalman_r))
        self.slider_r.valueChanged.connect(self.on_kalman_slider_changed)
        self.slider_r.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #475569;
                height: 6px;
                background: #1e293b;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ec4899;
                border: 1px solid #be185d;
                width: 14px;
                height: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        r_layout.addWidget(r_lbl)
        r_layout.addWidget(self.slider_r)
        r_layout.addWidget(self.r_val_lbl)
        layout.addLayout(r_layout)

        # Action Buttons Layout
        btn_layout = QHBoxLayout()
        self.btn_run_kalman = QPushButton("Run Kalman & Generate TRC")
        configure_button(self.btn_run_kalman, text="Run Kalman & Generate TRC", icon_name="refresh-cw", icon_color="#ffffff", bg_color="#a855f7")
        self.btn_run_kalman.clicked.connect(self.on_run_kalman_clicked)

        self.kalman_status_lbl = QLabel("")
        self.kalman_status_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;")

        btn_layout.addWidget(self.btn_run_kalman)
        btn_layout.addWidget(self.kalman_status_lbl)
        layout.addLayout(btn_layout)

        # Bottom Dialog Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        button_box.button(QDialogButtonBox.StandardButton.Ok).setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #10b981;
            }
        """)
        layout.addWidget(button_box)

    def q_to_slider(self, q):
        import numpy as np
        return int(np.clip(1 + (np.log10(max(1e-5, q)) - (-5)) * 99 / 4.0, 1, 100))

    def slider_to_q(self, val):
        return float(10 ** (-5.0 + (val - 1) * 4.0 / 99.0))

    def r_to_slider(self, r):
        import numpy as np
        return int(np.clip(1 + (np.log10(max(1e-4, r)) - (-4)) * 99 / 4.0, 1, 100))

    def slider_to_r(self, val):
        return float(10 ** (-4.0 + (val - 1) * 4.0 / 99.0))

    def on_trc_mode_changed(self, index):
        use_kalman = (index == 1)
        if self.parent_win and hasattr(self.parent_win, "on_kalman_mode_changed"):
            self.parent_win.on_kalman_mode_changed(use_kalman)

    def on_run_kalman_clicked(self):
        q = self.slider_to_q(self.slider_q.value())
        r = self.slider_to_r(self.slider_r.value())
        if self.parent_win and hasattr(self.parent_win, "run_and_save_kalman_trc"):
            self.kalman_status_lbl.setText("Generating TRC...")
            self.kalman_status_lbl.repaint()
            success = self.parent_win.run_and_save_kalman_trc(q, r)
            if success:
                self.combo_trc_mode.blockSignals(True)
                self.combo_trc_mode.setCurrentIndex(1)
                self.combo_trc_mode.blockSignals(False)
                self.kalman_status_lbl.setText("Generated successfully!")
            else:
                self.kalman_status_lbl.setText("Generation error.")

    def on_kalman_slider_changed(self, _value):
        q = self.slider_to_q(self.slider_q.value())
        r = self.slider_to_r(self.slider_r.value())
        self.q_val_lbl.setText(f"{q:.5f}")
        self.r_val_lbl.setText(f"{r:.4f}")

