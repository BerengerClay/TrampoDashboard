import numpy as np
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QMenu, QSplitter
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, QTimer
from acrobatics import calculate_acrobatics_summary, format_fig_trampoline_code
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from constants import KEYPOINT_COLORS, COCO_SKELETON
from icons import get_lucide_icon, configure_button


def log_debug(msg):
    try:
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with open("annotator.log", "a", encoding="utf-8") as f:
            f.write(f"[{now}] [3D_Visualizer] {msg}\n")
            f.flush()
    except Exception:
        pass

class Visualizer3DWidget(QWidget):
    """Matplotlib-based 3D skeleton visualizer widget that can be used inline or inside a window."""
    def __init__(self, main_win, parent=None, small_mode=False):
        super().__init__(parent)
        self.main_win = main_win
        self.small_mode = small_mode
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Matplotlib Figure and Canvas
        self.figure = Figure(facecolor='#090d16')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 3D Axes
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#090d16')
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Style grid/panes of 3D plot
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1e293b')
        self.ax.yaxis.pane.set_edgecolor('#1e293b')
        self.ax.zaxis.pane.set_edgecolor('#1e293b')
        self.ax.grid(True, color='#334155', linestyle='--')
        
        if self.small_mode:
            self.ax.tick_params(colors='#94a3b8', labelsize=7, pad=1)
            
            # Tiny popout button in the corner of the small visualizer
            self.btn_popout = QPushButton(self)
            self.btn_popout.setIcon(get_lucide_icon("external-link", color="#38bdf8"))
            self.btn_popout.setStyleSheet("background-color: rgba(15, 23, 42, 200); border: 1px solid #334155; padding: 2px; border-radius: 4px;")
            self.btn_popout.clicked.connect(self.main_win.toggle_3d_window)
        else:
            self.ax.set_xlabel('X (m)', color='#94a3b8')
            self.ax.set_ylabel('Y (m)', color='#94a3b8')
            self.ax.set_zlabel('Z (m)', color='#94a3b8')
            self.ax.tick_params(colors='#94a3b8')
            
        self.view_mode = "athlete" if self.small_mode else "global"
        self.keypoint_size = getattr(self.main_win, "keypoint_size_3d", 14) if self.main_win else 14
        self.update_plot(None)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'btn_popout') and self.btn_popout:
            self.btn_popout.resize(24, 24)
            self.btn_popout.move(self.width() - 28, 4)

    def update_plot(self, pts_3d, gt_pts_3d=None):
        """Updates the 3D scatter and line plots with new 3D keypoint coordinates."""
        log_debug(f"Visualizer3DWidget.update_plot started (small_mode={self.small_mode})")
        self.ax.cla()
        self.ax.set_facecolor('#090d16')
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1e293b')
        self.ax.yaxis.pane.set_edgecolor('#1e293b')
        self.ax.zaxis.pane.set_edgecolor('#1e293b')
        self.ax.grid(True, color='#334155', linestyle='--')
        
        if self.small_mode:
            self.ax.tick_params(colors='#94a3b8', labelsize=7, pad=1)
        else:
            self.ax.set_xlabel('X (m)', color='#94a3b8')
            self.ax.set_ylabel('Y (m)', color='#94a3b8')
            self.ax.set_zlabel('Z (m)', color='#94a3b8')
            self.ax.tick_params(colors='#94a3b8')
            
        if pts_3d is None or np.all(np.isnan(pts_3d)):
            self.ax.text2D(0.5, 0.5, "Not enough points\n(triangulate to generate 3D)", 
                           color='#94a3b8', ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return
            
        # Transform coords for axis mapping (inverse zup2yup rotation)
        pts_3d = pts_3d.copy()
        pts_3d_mapped = np.zeros_like(pts_3d)
        pts_3d_mapped[:, 0] = pts_3d[:, 2] # X = Z_trc
        pts_3d_mapped[:, 1] = pts_3d[:, 0] # Y = X_trc
        pts_3d_mapped[:, 2] = pts_3d[:, 1] # Z = Y_trc
        pts_3d = pts_3d_mapped

        # Draw skeleton lines with matching segment colors
        for conn in COCO_SKELETON:
            p1, p2 = conn
            pt1 = pts_3d[p1]
            pt2 = pts_3d[p2]
            
            if not np.isnan(pt1[0]) and not np.isnan(pt2[0]):
                if conn in [(5, 6), (11, 12)]:
                    col_str = '#10b981'
                elif conn in [(0, 1), (0, 2), (1, 3), (2, 4)]:
                    col_str = '#ec4899'
                elif p1 in [5, 7, 9, 11, 13, 15] and p2 in [5, 7, 9, 11, 13, 15]:
                    col_str = '#06b6d4'
                else:
                    col_str = '#f97316'
                    
                self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                             color=col_str, linewidth=1.5 if self.small_mode else 2, zorder=2)

        # Draw joints
        xs, ys, zs = [], [], []
        colors = []
        
        for idx in range(17):
            pt = pts_3d[idx]
            if not np.isnan(pt[0]):
                xs.append(pt[0])
                ys.append(pt[1])
                zs.append(pt[2])
                qcol = KEYPOINT_COLORS.get(idx, QColor(0, 255, 0))
                colors.append([qcol.red()/255.0, qcol.green()/255.0, qcol.blue()/255.0])
                
        if xs:
            kp_size = self.main_win.keypoint_size_3d if (self.main_win and hasattr(self.main_win, 'keypoint_size_3d')) else getattr(self, 'keypoint_size', 14)
            self.ax.scatter(xs, ys, zs, c=colors, s=kp_size if not self.small_mode else kp_size * 0.6, depthshade=True, zorder=10)

        # Draw 3D Ground Truth (GT) skeleton if enabled and available
        if self.main_win and getattr(self.main_win, "show_gt_overlay", False):
            if gt_pts_3d is None and hasattr(self.main_win, "calculate_gt_3d_keypoints"):
                gt_pts_3d = self.main_win.calculate_gt_3d_keypoints()
            if gt_pts_3d is not None and not np.all(np.isnan(gt_pts_3d)):
                gt_pts_3d = gt_pts_3d.copy()
                gt_pts_3d_mapped = np.zeros_like(gt_pts_3d)
                gt_pts_3d_mapped[:, 0] = gt_pts_3d[:, 2] # X = Z_trc
                gt_pts_3d_mapped[:, 1] = gt_pts_3d[:, 0] # Y = X_trc
                gt_pts_3d_mapped[:, 2] = gt_pts_3d[:, 1] # Z = Y_trc
                gt_pts_3d = gt_pts_3d_mapped

                gt_xs, gt_ys, gt_zs = [], [], []
                for idx in range(17):
                    pt = gt_pts_3d[idx]
                    if not np.isnan(pt[0]):
                        gt_xs.append(pt[0])
                        gt_ys.append(pt[1])
                        gt_zs.append(pt[2])

                for conn in COCO_SKELETON:
                    p1, p2 = conn
                    pt1 = gt_pts_3d[p1]
                    pt2 = gt_pts_3d[p2]
                    if not np.isnan(pt1[0]) and not np.isnan(pt2[0]):
                        self.ax.plot(
                            [pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]],
                            color='#fbbf24', linestyle='--', linewidth=2.5 if not self.small_mode else 1.8, zorder=8
                        )

                if gt_xs:
                    kp_size = self.main_win.keypoint_size_3d if (self.main_win and hasattr(self.main_win, 'keypoint_size_3d')) else getattr(self, 'keypoint_size', 14)
                    self.ax.scatter(
                        gt_xs, gt_ys, gt_zs,
                        c='#f59e0b', s=kp_size * 1.1 if not self.small_mode else kp_size * 0.7,
                        depthshade=False, zorder=12
                    )
                             
        # Set 3D axes limits
        if self.view_mode == "global" and self.main_win and getattr(self.main_win, 'global_3d_bounds', None) is not None:
            raw_bounds = self.main_win.global_3d_bounds
            bounds = {
                "x_min": raw_bounds["z_min"], "x_max": raw_bounds["z_max"],
                "y_min": raw_bounds["x_min"], "y_max": raw_bounds["x_max"],
                "z_min": raw_bounds["y_min"], "z_max": raw_bounds["y_max"],
            }
            max_range = max(bounds["x_max"] - bounds["x_min"], 
                            bounds["y_max"] - bounds["y_min"], 
                            bounds["z_max"] - bounds["z_min"])
            if max_range == 0:
                max_range = 1.0
            mid_x = (bounds["x_max"] + bounds["x_min"]) * 0.5
            mid_y = (bounds["y_max"] + bounds["y_min"]) * 0.5
            mid_z = (bounds["z_max"] + bounds["z_min"]) * 0.5
            
            self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
            self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
            self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        else:
            all_x = pts_3d[~np.isnan(pts_3d[:, 0]), 0]
            all_y = pts_3d[~np.isnan(pts_3d[:, 1]), 1]
            all_z = pts_3d[~np.isnan(pts_3d[:, 2]), 2]
            
            if len(all_x) > 0:
                max_range = max(all_x.max() - all_x.min(), 
                                all_y.max() - all_y.min(), 
                                all_z.max() - all_z.min())
                if max_range == 0:
                    max_range = 1.0
                mid_x = (all_x.max() + all_x.min()) * 0.5
                mid_y = (all_y.max() + all_y.min()) * 0.5
                mid_z = (all_z.max() + all_z.min()) * 0.5
                
                self.ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
                self.ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
                self.ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)

        self.canvas.draw_idle()


class AcrobaticsChartWidget(QWidget):
    """Matplotlib-based 2D chart widget displaying Somersaults (Salto), Twists (Vrille), Impacts, and Postures (Tuck/Pike/Straight)."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(facecolor='#090d16')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.91)
        
        self.coords_3d = None
        self.saltos_per_jump = None
        self.vrilles_per_jump = None
        self.saltos_cumul = None
        self.vrilles_cumul = None
        self.acro_results = None
        self.num_frames = 0
        self.current_frame = 0
        self.vline = None
        self.display_mode = "per_jump" # "per_jump" or "cumulative"
        self.fig_mode = True # Snapping to official FIG multiples (0.5 steps)
        
        self.draw_empty()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #a855f7;
                color: white;
            }
        """)
        action_fig = menu.addAction("Official FIG Normalization (3.0, 2.0, 1.5 somersaults)")
        action_fig.setCheckable(True)
        action_fig.setChecked(self.fig_mode)

        selected = menu.exec(event.globalPos())
        if selected == action_fig:
            self.fig_mode = not self.fig_mode
            self.plot_chart()

    def set_data(self, coords_3d):
        """Calculates acrobatic rotations and posture classifications for the sequence."""
        if coords_3d is None or len(coords_3d) == 0:
            self.coords_3d = None
            self.saltos_per_jump = None
            self.vrilles_per_jump = None
            self.saltos_cumul = None
            self.vrilles_cumul = None
            self.acro_results = None
            self.impacts = None
            self.num_frames = 0
            self.draw_empty()
            return
            
        self.coords_3d = coords_3d
        self.num_frames = len(coords_3d)
        
        try:
            (
                self.saltos_per_jump,
                self.vrilles_per_jump,
                self.saltos_cumul,
                self.vrilles_cumul,
                self.impacts,
                self.acro_results
            ) = calculate_acrobatics_summary(coords_3d)
        except Exception as e:
            print(f"[Acrobatics] Error computing kinematics: {e}")
            self.saltos_per_jump = np.zeros(self.num_frames)
            self.vrilles_per_jump = np.zeros(self.num_frames)
            self.saltos_cumul = np.zeros(self.num_frames)
            self.vrilles_cumul = np.zeros(self.num_frames)
            self.impacts = np.array([0, self.num_frames - 1])
            self.acro_results = [{"position": "Unknown", "base_deduction": 0, "knee_deduction": 0}] * self.num_frames
            
        self.plot_chart()

    def draw_empty(self):
        self.ax.cla()
        self.ax.set_facecolor('#090d16')
        self.ax.text(0.5, 0.5, "No Acrobatics Data\n(Load 3D Trajectory)",
                    color='#94a3b8', ha='center', va='center', transform=self.ax.transAxes)
        self.ax.tick_params(colors='#94a3b8')
        self.canvas.draw()

    def plot_chart(self):
        if self.saltos_per_jump is None or self.num_frames == 0:
            self.draw_empty()
            return
            
        self.ax.cla()
        self.ax.set_facecolor('#090d16')
        self.ax.grid(True, color='#1e293b', linestyle='--', alpha=0.7)
        
        t_frames = np.arange(self.num_frames)
        
        saltos_plot = self.saltos_per_jump.copy()
        vrilles_plot = self.vrilles_per_jump.copy()

        # FIG Official Normalization / Scaling if active in per-jump mode
        jump_badges = []
        if self.impacts is not None and len(self.impacts) >= 2:
            for k in range(len(self.impacts) - 1):
                s, e = self.impacts[k], self.impacts[k + 1]
                if e > s:
                    max_s = np.max(saltos_plot[s:e])
                    max_v = np.max(vrilles_plot[s:e])
                    
                    target_s = round(max_s / 0.5) * 0.5
                    target_v = round(max_v / 0.5) * 0.5
                    
                    if self.fig_mode:
                        if max_s > 0.3 and target_s > 0:
                            saltos_plot[s:e] = saltos_plot[s:e] * (target_s / max_s)
                        if max_v > 0.3 and target_v > 0:
                            vrilles_plot[s:e] = vrilles_plot[s:e] * (target_v / max_v)
                            
                    mid_f = (s + e) // 2
                    disp_s = target_s if self.fig_mode else round(max_s, 2)
                    disp_v = target_v if self.fig_mode else round(max_v, 2)

                    # Determine predominant posture in jump window using acrobatic threshold rule (>= 20%)
                    posture = "Straight"
                    if self.acro_results and s < len(self.acro_results):
                        total_f = max(1, e - s)
                        tuck_cnt = sum(1 for item in self.acro_results[s:e] if item.get("position") == "Tuck")
                        pike_cnt = sum(1 for item in self.acro_results[s:e] if item.get("position") == "Pike")
                        if tuck_cnt / total_f >= 0.20:
                            posture = "Tuck"
                        elif pike_cnt / total_f >= 0.20:
                            posture = "Pike"

                    # For multiple saltos (Double, Triple, etc.), measure twist accumulated at completion of each salto
                    vrilles_per_salto = []
                    n_s = max(1, int(round(target_s)))
                    if n_s >= 2 and e > s:
                        jump_s = saltos_plot[s:e]
                        jump_v = vrilles_plot[s:e]
                        for salto_idx in range(1, n_s):
                            indices = np.where(jump_s >= float(salto_idx))[0]
                            if len(indices) > 0:
                                vrilles_per_salto.append(jump_v[indices[0]])
                            else:
                                vrilles_per_salto.append(0.0)

                    fig_code = format_fig_trampoline_code(disp_s, disp_v, posture, vrilles_per_salto=vrilles_per_salto)

                    if max_s > 0.3 or max_v > 0.3:
                        jump_badges.append((mid_f, disp_s, disp_v, fig_code, k + 1))

        # Plot Salto (Somersault) & Vrille (Twist) curves
        self.ax.plot(t_frames, saltos_plot, label="Somersaults", color="#ef4444", lw=2)
        self.ax.plot(t_frames, vrilles_plot, label="Twists", color="#3b82f6", lw=2)
        
        # Display Jump Summary Badges at top
        max_y = max(3.2, np.max(saltos_plot) * 1.1) if len(saltos_plot) > 0 else 3.2
        for mid_f, disp_s, disp_v, fig_code, j_num in jump_badges:
            if self.fig_mode and fig_code:
                badge_str = f"Jump {j_num}: FIG {fig_code} ({disp_s:g}S | {disp_v:g}T)"
            else:
                badge_str = f"Jump {j_num}: {disp_s:g}S | {disp_v:g}T"
            self.ax.text(mid_f, max_y * 0.88, badge_str, color='#fbbf24', fontsize=8, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e293b', edgecolor='#fbbf24', alpha=0.95), zorder=6)
        
        # Milestone markers (+1/4 Somersault, +1/2 Twist)
        salto_cross_f, salto_cross_v = [], []
        vrille_cross_f, vrille_cross_v = [], []
        for i in range(1, self.num_frames):
            ps, cs = saltos_plot[i - 1], saltos_plot[i]
            pv, cv = vrilles_plot[i - 1], vrilles_plot[i]
            if ps <= cs and int(cs / 0.25) > int(ps / 0.25):
                salto_cross_f.append(t_frames[i])
                salto_cross_v.append(cs)
            if pv <= cv and int(cv / 0.5) > int(pv / 0.5):
                vrille_cross_f.append(t_frames[i])
                vrille_cross_v.append(cv)
                
        if salto_cross_f:
            self.ax.scatter(salto_cross_f, salto_cross_v, color="#ef4444", edgecolors="white", s=30, label="+1/4 Somersault", zorder=4)
        if vrille_cross_f:
            self.ax.scatter(vrille_cross_f, vrille_cross_v, color="#3b82f6", edgecolors="white", s=30, marker="s", label="+1/2 Twist", zorder=4)
            
        # Draw trampoline impact lines (purple dashed)
        if self.impacts is not None and len(self.impacts) > 2:
            inner_impacts = self.impacts[1:-1]
            for peak in inner_impacts:
                self.ax.axvline(peak, color="#a855f7", linestyle="--", alpha=0.6, lw=1.5, zorder=3)
            self.ax.plot([], [], color="#a855f7", linestyle="--", alpha=0.6, label="Impact")

        # Background posture shading
        pos_colors = {"Tuck": "#22c55e", "Pike": "#f43f5e", "Straight": "#38bdf8"}
        added_labels = set()
        if self.acro_results and len(self.acro_results) == self.num_frames:
            start_idx = 0
            curr_pos = self.acro_results[0]["position"]
            for i in range(1, self.num_frames):
                pos = self.acro_results[i]["position"]
                is_last = (i == self.num_frames - 1)
                if pos != curr_pos or is_last:
                    end_idx = i if pos != curr_pos else i + 1
                    if curr_pos in pos_colors:
                        lbl = curr_pos if curr_pos not in added_labels else None
                        if lbl:
                            added_labels.add(curr_pos)
                        self.ax.axvspan(start_idx, end_idx - 1, color=pos_colors[curr_pos], alpha=0.15, lw=0, label=lbl)
                    curr_pos = pos
                    start_idx = i
                    
        fig_title = "Acrobatics Rotations (Per-Jump)"
        if self.fig_mode:
            fig_title += " - Official FIG Normalization"
        self.ax.set_title(fig_title, color='#f8fafc', fontsize=10, fontweight='bold')
        self.ax.set_xlabel("Frame Index", color='#94a3b8', fontsize=9)
        self.ax.set_ylabel("Rotations (Turns)", color='#94a3b8', fontsize=9)
        self.ax.tick_params(colors='#94a3b8', labelsize=8)
        self.ax.set_xlim(0, max(1, self.num_frames - 1))
        self.ax.set_ylim(-0.1, max_y)
        
        self.ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncols=6,
            frameon=True,
            facecolor='#1e293b',
            edgecolor='#334155',
            labelcolor='#f8fafc',
            fontsize=8
        )
        
        # Cursor vertical line
        self.vline = self.ax.axvline(self.current_frame, color='#f43f5e', linestyle=':', linewidth=1.5, zorder=5)
        self.canvas.draw_idle()

    def update_frame_cursor(self, frame_idx):
        self.current_frame = frame_idx
        if hasattr(self, 'vline') and self.vline:
            self.vline.set_xdata([frame_idx, frame_idx])
            self.canvas.draw_idle()


class Visualizer3DWindow(QMainWindow):
    """Separate window container for the 3D visualizer that adds play/pause, sync, acrobatics chart and progress controls."""
    def __init__(self, main_win, parent=None):
        super().__init__(parent)
        self.main_win = main_win
        self.setWindowTitle("3D Skeleton & Acrobatics Rotations Visualizer")
        self.resize(1150, 650)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Splitter containing 3D canvas on the left & 2D Acrobatics chart on the right
        canvas_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        self.widget_3d = Visualizer3DWidget(main_win, self, small_mode=False)
        self.widget_acro_chart = AcrobaticsChartWidget(main_win, self)
        
        canvas_splitter.addWidget(self.widget_3d)
        canvas_splitter.addWidget(self.widget_acro_chart)
        canvas_splitter.setSizes([380, 770])
        
        main_layout.addWidget(canvas_splitter, stretch=1)
        
        # Playback control panel at the bottom
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(5, 5, 5, 5)
        
        self.btn_play_pause = QPushButton()
        configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        
        self.btn_prev = QPushButton()
        configure_button(self.btn_prev, text="Prev", icon_name="arrow-left", icon_color="#ffffff")
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_prev.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155;")
        
        self.btn_next = QPushButton()
        configure_button(self.btn_next, text="Next", icon_name="arrow-right", icon_color="#ffffff")
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_next.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155;")

        # Playback speed options
        self.speed_options = [0.25, 0.5, 1.0, 2.0, 4.0]
        self.current_speed_idx = 2
        
        self.btn_speed = QPushButton()
        configure_button(self.btn_speed, text="x1", icon_name="gauge", icon_color="#38bdf8")
        self.btn_speed.setToolTip("Playback Speed (Click to cycle, right-click to select)")
        self.btn_speed.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; font-weight: bold;")
        self.btn_speed.clicked.connect(self.cycle_speed)
        self.btn_speed.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_speed.customContextMenuRequested.connect(self.show_speed_menu)

        self.btn_view_mode = QPushButton()
        configure_button(self.btn_view_mode, text="Global Mode", icon_name="globe", icon_color="#ffffff")
        self.btn_view_mode.clicked.connect(self.toggle_view_mode)
        self.btn_view_mode.setToolTip("Toggle between Global View and Athlete Focus Mode")
        
        self.btn_trc_mode = QPushButton()
        self.btn_trc_mode.setToolTip("Toggle between raw 3D trajectory (triangulated.trc) and smoothed (triangulated_kalman.trc)")
        self.btn_trc_mode.clicked.connect(self.toggle_trc_mode)
        self.update_trc_mode_button()

        self.btn_toggle_chart = QPushButton()
        configure_button(self.btn_toggle_chart, text="Chart 2D", icon_name="bar-chart-2", icon_color="#ffffff")
        self.btn_toggle_chart.setToolTip("Show/Hide 2D Acrobatics Rotations Chart to optimize animation FPS")
        self.btn_toggle_chart.clicked.connect(self.toggle_chart_visibility)

        self.playback_slider = QSlider(Qt.Orientation.Horizontal)
        self.playback_slider.valueChanged.connect(self.on_slider_moved)
        self.playback_slider.setStyleSheet("""
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
        
        self.lbl_frame_info = QLabel("Frame: 0 / 0")
        self.lbl_frame_info.setStyleSheet("color: #94a3b8; font-weight: bold; min-width: 90px;")
        
        control_layout.addWidget(self.btn_play_pause)
        control_layout.addWidget(self.btn_prev)
        control_layout.addWidget(self.btn_next)
        control_layout.addWidget(self.btn_speed)
        control_layout.addWidget(self.btn_trc_mode)
        control_layout.addWidget(self.btn_view_mode)
        control_layout.addWidget(self.btn_toggle_chart)
        control_layout.addWidget(self.playback_slider)
        control_layout.addWidget(self.lbl_frame_info)
        
        main_layout.addWidget(control_panel)
        
        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)
        self.playback_frame_idx = 0
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }
            QWidget {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)

    def cycle_speed(self):
        self.current_speed_idx = (self.current_speed_idx + 1) % len(self.speed_options)
        speed = self.speed_options[self.current_speed_idx]
        self.set_speed(speed)

    def toggle_trc_mode(self):
        """Toggles between raw and Kalman-smoothed 3D TRC trajectories."""
        if not self.main_win:
            return
        new_mode = not getattr(self.main_win, "use_kalman_trc", False)
        if hasattr(self.main_win, "on_kalman_mode_changed"):
            self.main_win.on_kalman_mode_changed(new_mode)
        self.update_trc_mode_button()

    def update_trc_mode_button(self):
        """Updates TRC mode button styling and label."""
        if not self.main_win:
            return
        use_kalman = getattr(self.main_win, "use_kalman_trc", False)
        if use_kalman:
            configure_button(self.btn_trc_mode, text="TRC: Kalman Filtered", icon_name="activity", icon_color="#ffffff", bg_color="#a855f7")
        else:
            configure_button(self.btn_trc_mode, text="TRC: Raw (No Kalman)", icon_name="activity", icon_color="#ffffff", bg_color="#475569")

    def set_speed(self, speed):
        if speed in self.speed_options:
            self.current_speed_idx = self.speed_options.index(speed)
        else:
            self.speed_options.append(speed)
            self.speed_options.sort()
            self.current_speed_idx = self.speed_options.index(speed)
        
        speed_str = f"x{speed:g}"
        configure_button(self.btn_speed, text=speed_str, icon_name="gauge", icon_color="#38bdf8")
        
        if self.play_timer.isActive():
            base_fps = 30.0
            interval = max(1, int(1000.0 / (base_fps * speed)))
            self.play_timer.start(interval)

    def show_speed_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #334155;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #3b82f6;
                color: #ffffff;
            }
        """)
        for spd in self.speed_options:
            spd_str = f"x{spd:g}"
            action = menu.addAction(spd_str)
            if spd == self.speed_options[self.current_speed_idx]:
                action.setCheckable(True)
                action.setChecked(True)
            action.triggered.connect(lambda checked, s=spd: self.set_speed(s))
        menu.exec(self.btn_speed.mapToGlobal(pos))

    def toggle_playback(self):
        if not self.main_win or not self.main_win.sorted_frames:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
            configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        else:
            speed = self.speed_options[self.current_speed_idx]
            base_fps = 30.0
            interval = max(1, int(1000.0 / (base_fps * speed)))
            self.play_timer.start(interval)
            configure_button(self.btn_play_pause, text="Pause", icon_name="pause", icon_color="#ffffff", bg_color="#d97706")

    def update_fps(self, fps):
        if self.play_timer.isActive():
            speed = self.speed_options[self.current_speed_idx]
            interval = max(1, int(1000.0 / max(1, fps * speed)))
            self.play_timer.start(interval)

    def advance_frame(self):
        if not self.main_win or not self.main_win.sorted_frames:
            self.play_timer.stop()
            return
        total = len(self.main_win.sorted_frames)
        self.playback_frame_idx = (self.playback_frame_idx + 1) % total
        
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(self.playback_frame_idx)
        self.playback_slider.blockSignals(False)
        
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def on_slider_moved(self, value):
        self.playback_frame_idx = value
        total = len(self.main_win.sorted_frames) if (self.main_win and self.main_win.sorted_frames) else 0
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def prev_frame(self):
        if not self.main_win or not self.main_win.sorted_frames:
            return
        if self.play_timer.isActive():
            self.toggle_playback()
        total = len(self.main_win.sorted_frames)
        self.playback_frame_idx = (self.playback_frame_idx - 1) % total
        
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(self.playback_frame_idx)
        self.playback_slider.blockSignals(False)
        
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def next_frame(self):
        if not self.main_win or not self.main_win.sorted_frames:
            return
        if self.play_timer.isActive():
            self.toggle_playback()
        total = len(self.main_win.sorted_frames)
        self.playback_frame_idx = (self.playback_frame_idx + 1) % total
        
        self.playback_slider.blockSignals(True)
        self.playback_slider.setValue(self.playback_frame_idx)
        self.playback_slider.blockSignals(False)
        
        self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
        self.update_visualization()

    def sync_to_annotator_frame(self):
        log_debug("Visualizer3DWindow.sync_to_annotator_frame started")
        if self.play_timer.isActive():
            self.play_timer.stop()
            configure_button(self.btn_play_pause, text="Play", icon_name="play", icon_color="#ffffff", bg_color="#059669")
        if self.main_win and self.main_win.sorted_frames:
            self.playback_frame_idx = self.main_win.current_frame_idx
            
            self.playback_slider.blockSignals(True)
            self.playback_slider.setValue(self.playback_frame_idx)
            self.playback_slider.blockSignals(False)
            
            total = len(self.main_win.sorted_frames)
            self.lbl_frame_info.setText(f"Frame: {self.playback_frame_idx + 1} / {total}")
            
            if hasattr(self.main_win, "get_all_3d_coordinates"):
                all_coords = self.main_win.get_all_3d_coordinates()
                if all_coords is not None:
                    self.widget_acro_chart.set_data(all_coords)
        self.update_visualization()
        log_debug("Visualizer3DWindow.sync_to_annotator_frame completed")

    def toggle_view_mode(self):
        if self.widget_3d.view_mode == "athlete":
            self.widget_3d.view_mode = "global"
            configure_button(self.btn_view_mode, text="Global Mode", icon_name="globe", icon_color="#ffffff")
        else:
            self.widget_3d.view_mode = "athlete"
            configure_button(self.btn_view_mode, text="Focus Mode", icon_name="maximize", icon_color="#ffffff")
        self.update_visualization()

    def toggle_chart_visibility(self):
        """Shows or hides the 2D Acrobatics Rotations chart widget to optimize playback FPS."""
        vis = not self.widget_acro_chart.isVisible()
        self.widget_acro_chart.setVisible(vis)
        if vis:
            configure_button(self.btn_toggle_chart, text="Chart 2D", icon_name="bar-chart-2", icon_color="#ffffff")
        else:
            configure_button(self.btn_toggle_chart, text="Chart 2D (Off)", icon_name="bar-chart-2", icon_color="#94a3b8", bg_color="#334155")
        self.update_visualization()

    def update_visualization(self):
        log_debug("Visualizer3DWindow.update_visualization started")
        if self.main_win:
            pts_3d = self.main_win.calculate_3d_keypoints(self.playback_frame_idx)
            gt_pts_3d = self.main_win.calculate_gt_3d_keypoints(self.playback_frame_idx)
            log_debug("Visualizer3DWindow.update_visualization calculate_3d_keypoints done")
            self.widget_3d.update_plot(pts_3d, gt_pts_3d=gt_pts_3d)
            log_debug("Visualizer3DWindow.update_visualization update_plot done")

            # Update 2D chart cursor only if chart is visible
            if self.widget_acro_chart.isVisible():
                if self.widget_acro_chart.saltos_per_jump is None and hasattr(self.main_win, "get_all_3d_coordinates"):
                    all_coords_3d = self.main_win.get_all_3d_coordinates()
                    if all_coords_3d is not None:
                        self.widget_acro_chart.set_data(all_coords_3d)
                        
                # Throttle 2D cursor redraws during active video playback for maximum FPS
                if not self.play_timer.isActive() or (self.playback_frame_idx % 2 == 0):
                    self.widget_acro_chart.update_frame_cursor(self.playback_frame_idx)

    def closeEvent(self, event):
        self.play_timer.stop()
        super().closeEvent(event)
