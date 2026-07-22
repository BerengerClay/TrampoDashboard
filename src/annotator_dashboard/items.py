from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsRectItem
from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath, QPainterPathStroker
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer

from constants import KEYPOINT_COLORS

class KeypointItem(QGraphicsEllipseItem):
    """Interactive keypoint dot that updates positions in real-time when dragged."""
    def __init__(self, x, y, point_id, name, parent_widget, kv):
        radius = parent_widget.main_win.keypoint_radius if parent_widget and parent_widget.main_win else 6
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.point_id = point_id
        self.name = name
        self.parent_widget = parent_widget
        self.kv = kv
        
        # Color based on joint type (fully opaque, keep current colors)
        color = KEYPOINT_COLORS.get(point_id, QColor(0, 255, 0))
        self.setBrush(QBrush(color))
        self.update_pen(radius)
        
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsEllipseItem.GraphicsItemFlag.ItemIsFocusable)
        self.setAcceptHoverEvents(True)
        conf_str = "Manual" if self.kv >= 2.0 else f"{self.kv:.2f}"
        self.setToolTip(f"{name} (ID: {point_id}, Conf: {conf_str})")
        self.setZValue(5.0)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        
        # Group dragging states
        self._is_dragging_group = False
        self._selected_kps_to_drag = []
        self._selected_bboxes_to_drag = []
        self._drag_start_positions = {}
        self._drag_start_scene = QPointF()
 
    def update_pen(self, radius):
        """Scale border thickness and interpolate color between black and white based on confidence."""
        pen_width = max(1.0, radius / 4.0)
        
        show_conf = False
        if self.parent_widget and self.parent_widget.main_win:
            show_conf = getattr(self.parent_widget.main_win, "vitpose_show_confidence", True)
            
        if show_conf and self.kv <= 1.0:
            # Interpolate contour color between black (0) and white (255) based on self.kv
            c_val = int(max(0.0, min(1.0, self.kv)) * 255)
            self.setPen(QPen(QColor(c_val, c_val, c_val), pen_width))
        elif self.kv == 1:
            self.setPen(QPen(QColor(234, 179, 8), pen_width, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(Qt.GlobalColor.white, pen_width))
 
    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionChange and self.parent_widget:
            new_pos = value
            # Notify the parent widget of the manual position adjustment (do not save/sync to disk/3D yet)
            self.parent_widget.update_keypoint_pos(self.point_id, new_pos.x(), new_pos.y(), save_and_sync=False)
            
            # Transition visibility to 2 (manual reference) immediately on drag
            if self.kv <= 1.0:
                self.kv = 2.0
                radius = self.rect().width() / 2.0
                self.update_pen(radius)
                self.setToolTip(f"{self.name} (ID: {self.point_id}, Conf: Manual)")
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        event.ignore()

    def hoverEnterEvent(self, event):
        # Always display hovered point name in status bar, even when Delete key is held
        if self.parent_widget and self.parent_widget.main_win:
            conf_str = "Manual" if self.kv >= 2.0 else f"{self.kv:.2f}"
            self.parent_widget.main_win.status_bar.showMessage(
                f"Hovered Joint: {self.name} (ID: {self.point_id}, Conf: {conf_str})"
            )
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        if self.parent_widget and self.parent_widget.main_win:
            self.parent_widget.main_win.status_bar.clearMessage()
        super().hoverLeaveEvent(event)

    def delete_point(self):
        if self.parent_widget:
            self.parent_widget.delete_keypoint(self.point_id)

    def set_radius(self, radius):
        """Update keypoint circle size and border thickness."""
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.update_pen(radius)


class SkeletonItem(QGraphicsLineItem):
    """Dynamic connection line between two KeypointItems."""
    def __init__(self, kp1, kp2, color):
        super().__init__()
        self.kp1 = kp1
        self.kp2 = kp2
        self.setPen(QPen(color, 2))  # Colored pen
        self.setZValue(3.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)
        self.update_position()

    def update_position(self):
        if self.kp1 and self.kp2:
            self.setLine(self.kp1.pos().x(), self.kp1.pos().y(),
                         self.kp2.pos().x(), self.kp2.pos().y())


class BBoxItem(QGraphicsRectItem):
    """Draggable and resizable bounding box item."""
    def __init__(self, rect, parent_widget):
        super().__init__(rect)
        self.parent_widget = parent_widget
        
        # Style bounding box
        self.setPen(QPen(QColor(234, 179, 8), 2)) # Yellow
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        # Make item read-only
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        self.setAcceptHoverEvents(False)
        self.handle_size = 0
        self.handles = {} # handle_position_name -> rect
        self.active_handle = None
        self.setZValue(1.0)
        self.update_handles()

        # Group dragging states
        self._selected_kps_to_drag = []
        self._selected_bboxes_to_drag = []
        self._drag_start_positions = {}

    def delete_bbox(self):
        # Save updated bounding box to memory database as empty [0,0,0,0]
        bbox_coords = [0.0, 0.0, 0.0, 0.0]
        # Defer execution to avoid deleting self within event handler
        QTimer.singleShot(0, lambda: self.parent_widget.main_win.update_bbox(self.parent_widget.camera_id, bbox_coords, preserve_view=False))

    def boundingRect(self):
        r = self.rect()
        s = self.handle_size
        hs = s / 2
        margin = 2 # Safety margin for pen thickness
        # Extend bounding box to include the resizing handles and pen borders
        return QRectF(r.left() - hs - margin, r.top() - hs - margin, r.width() + s + 2 * margin, r.height() + s + 2 * margin)

    def shape(self):
        """Define click hitbox shape to outline border and handles only, ignoring interior."""
        path = QPainterPath()
        
        # 1. outline borders with a thick stroke (16px click hitbox width)
        r = self.rect()
        border_path = QPainterPath()
        border_path.addRect(r)
        
        stroker = QPainterPathStroker()
        stroker.setWidth(6.0) # Click hitbox thickness
        stroked_border = stroker.createStroke(border_path)
        path.addPath(stroked_border)
        
        # 2. Add resize handles hitboxes
        for handle_rect in self.handles.values():
            path.addRect(handle_rect)
            
        return path

    def update_handles(self):
        self.prepareGeometryChange()
        r = self.rect()
        s = self.handle_size
        hs = s / 2
        
        # Top-left, Top-right, Bottom-left, Bottom-right corners
        self.handles = {
            "top_left": QRectF(r.left() - hs, r.top() - hs, s, s),
            "top_right": QRectF(r.right() - hs, r.top() - hs, s, s),
            "bottom_left": QRectF(r.left() - hs, r.bottom() - hs, s, s),
            "bottom_right": QRectF(r.right() - hs, r.bottom() - hs, s, s)
        }

    def paint(self, painter, option, widget):
        if self.isSelected():
            self.setPen(QPen(QColor(234, 179, 8), 2, Qt.PenStyle.DashLine))
        else:
            self.setPen(QPen(QColor(234, 179, 8), 2))
            
        super().paint(painter, option, widget)
        
        # Draw resize handles if selected or hovered
        painter.setPen(QPen(QColor(234, 179, 8), 1))
        painter.setBrush(QBrush(QColor(234, 179, 8)))
        for handle_rect in self.handles.values():
            painter.drawRect(handle_rect)

    def get_element_at_pos(self, pos):
        """Returns the corner handle name or edge name at the given position, or None."""
        # 1. Check corner handles first
        for name, handle_rect in self.handles.items():
            if handle_rect.contains(pos):
                return name
                
        # 2. Check edges if it's close to the border
        r = self.rect()
        px, py = pos.x(), pos.y()
        margin = 4.0 # Click margin thickness
        
        # Check top edge
        if abs(py - r.top()) <= margin and r.left() <= px <= r.right():
            return "top_edge"
        # Check bottom edge
        if abs(py - r.bottom()) <= margin and r.left() <= px <= r.right():
            return "bottom_edge"
        # Check left edge
        if abs(px - r.left()) <= margin and r.top() <= py <= r.bottom():
            return "left_edge"
        # Check right edge
        if abs(px - r.right()) <= margin and r.top() <= py <= r.bottom():
            return "right_edge"
            
        return None

    def update_cursor_shape(self, pos, modifiers):
        """Helper to set cursor shape without calling hoverMoveEvent with mouse event."""
        element = self.active_handle if self.active_handle else self.get_element_at_pos(pos)
        ctrl_pressed = modifiers & Qt.KeyboardModifier.ControlModifier
        
        if element is None:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif ctrl_pressed:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            angle = getattr(self.parent_widget, "current_rotation_angle", 0.0)
            angle = int(round(angle)) % 360
            
            if element in ("top_left", "bottom_right"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif element in ("top_right", "bottom_left"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            elif element in ("top_edge", "bottom_edge"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeHorCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif element in ("left_edge", "right_edge"):
                if angle in (90, 270):
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeHorCursor)

    def hoverMoveEvent(self, event):
        pos = event.pos()
        element = self.get_element_at_pos(pos)
        self.active_handle = element
        self.update_cursor_shape(pos, event.modifiers())
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        event.ignore()

    def mouseMoveEvent(self, event):
        event.ignore()

    def mouseReleaseEvent(self, event):
        event.ignore()


class ReprojectedPointItem(QGraphicsEllipseItem):
    """Hollow overlay circle showing the reprojected 3D coordinate for comparison."""
    def __init__(self, x, y, point_id, name, parent_widget, color=None):
        radius = parent_widget.main_win.keypoint_radius if parent_widget and parent_widget.main_win else 6
        super().__init__(-radius, -radius, radius * 2, radius * 2)
        self.setPos(x, y)
        self.point_id = point_id
        self.name = name
        self.parent_widget = parent_widget
        
        if color is None:
            color = QColor(244, 63, 94, 80)
        self.color = color
        
        pen_width = max(1.0, radius / 4.0)
        self.setPen(QPen(self.color, pen_width, Qt.PenStyle.DashLine))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        
        # Purely visual overlay, no interaction
        self.setFlags(QGraphicsEllipseItem.GraphicsItemFlag(0))
        self.setAcceptHoverEvents(False)
        self.setToolTip(f"Reprojected {name} (ID: {point_id})")
        self.setZValue(4.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)

    def set_radius(self, radius):
        """Update reprojected circle size and border thickness."""
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        pen_width = max(1.0, radius / 4.0)
        self.setPen(QPen(getattr(self, "color", QColor(244, 63, 94, 80)), pen_width, Qt.PenStyle.DashLine))


class DiscrepancyLineItem(QGraphicsLineItem):
    """Dashed connection line between the user's manual keypoint and the 3D reprojected coordinate."""
    def __init__(self, x1, y1, x2, y2, color=None):
        super().__init__(x1, y1, x2, y2)
        if color is None:
            color = QColor(244, 63, 94, 80)
        self.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
        self.setFlags(QGraphicsLineItem.GraphicsItemFlag(0))
        self.setZValue(2.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setEnabled(False)

