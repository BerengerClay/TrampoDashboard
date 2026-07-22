import os
import sys
import argparse
from PyQt6.QtWidgets import QApplication

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mainwindow import TrampolineAnnotator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-View Trampoline Jumper Annotator")
    parser.add_argument("paths", nargs="*", default=[], help="Path to sequence directory or 8 camera folders")
    parser.add_argument("--gt", "--gt-path", dest="gt_path", default=None, help="Path to Ground Truth file (.trc, .json, or .pkl)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TrampolineAnnotator(paths=args.paths, gt_path=args.gt_path)
    window.show()
    sys.exit(app.exec())
