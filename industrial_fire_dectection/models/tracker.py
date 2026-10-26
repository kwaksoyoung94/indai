import itertools

from config import Config


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """탐지 박스를 프레임 간 연결해 객체별(track_id) 시퀀스를 만든다.

    탐지가 일시적으로 끊겨도 max_age 프레임 동안은 마지막 박스를 유지하므로
    ROI 시퀀스가 시간적으로 연속되고, 서로 다른 객체가 섞이지 않는다.
    """

    def __init__(self, iou_thr=None, max_age=None):
        self.iou_thr = Config.TRACK_IOU if iou_thr is None else iou_thr
        self.max_age = Config.TRACK_MAX_AGE if max_age is None else max_age
        self._ids = itertools.count(1)
        self.tracks = {}    # id -> {'box': (x1,y1,x2,y2), 'age': int}

    def update(self, boxes):
        """boxes: [(x1,y1,x2,y2), ...]. 반환: [(track_id, box)] (유지 중인 트랙 전체)."""
        pairs = sorted(
            ((iou(t['box'], b), tid, bi)
             for tid, t in self.tracks.items() for bi, b in enumerate(boxes)),
            reverse=True)
        used_t, used_b = set(), set()
        for score, tid, bi in pairs:
            if score < self.iou_thr:
                break
            if tid in used_t or bi in used_b:
                continue
            self.tracks[tid] = {'box': tuple(boxes[bi]), 'age': 0}
            used_t.add(tid)
            used_b.add(bi)

        for tid in list(self.tracks):
            if tid not in used_t:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]

        for bi, b in enumerate(boxes):
            if bi not in used_b:
                self.tracks[next(self._ids)] = {'box': tuple(b), 'age': 0}

        return [(tid, t['box']) for tid, t in self.tracks.items()]

    def active_ids(self):
        return set(self.tracks)
