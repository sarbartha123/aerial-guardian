import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


# ---------------- IOU ---------------- #
def iou_batch(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))

    area_a = (a[:, 2]-a[:,0])*(a[:,3]-a[:,1])
    area_b = (b[:, 2]-b[:,0])*(b[:,3]-b[:,1])

    x1 = np.maximum(a[:,None,0], b[None,:,0])
    y1 = np.maximum(a[:,None,1], b[None,:,1])
    x2 = np.minimum(a[:,None,2], b[None,:,2])
    y2 = np.minimum(a[:,None,3], b[None,:,3])

    inter = np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    union = area_a[:,None]+area_b[None,:]-inter

    return inter/np.maximum(union,1e-6)


# ---------------- HUNGARIAN ---------------- #
def hungarian(cost, thresh):
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    r,c = linear_sum_assignment(-cost)
    matches = [(i,j) for i,j in zip(r,c) if cost[i,j] >= thresh]

    mr = {i for i,_ in matches}
    mc = {j for _,j in matches}

    ur = [i for i in range(cost.shape[0]) if i not in mr]
    uc = [j for j in range(cost.shape[1]) if j not in mc]

    return matches, ur, uc


# ---------------- FEATURE SIM ---------------- #
def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6))


# ---------------- HOMOGRAPHY ---------------- #
def estimate_homography(prev, curr):
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    pts1 = cv2.goodFeaturesToTrack(prev_g, 400, 0.01, 8)
    if pts1 is None:
        return None

    pts2, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, pts1, None)
    valid = st.flatten() == 1

    pts1 = pts1[valid]
    pts2 = pts2[valid]

    if len(pts1) < 30:
        return None

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    return H


# ================= TRACK ================= #
class Track:
    count = 0

    def __init__(self, bbox, feat=None, frame_id=0):
        Track.count += 1
        self.id = Track.count

        self.bbox = bbox[:4].copy()
        self.conf = bbox[4]

        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2

        self.center = np.array([cx,cy])
        self.velocity = np.array([0.0,0.0])

        self.feature = feat
        self.features = [feat] if feat is not None else []

        self.hits = 1
        self.time_since_update = 0
        self.last_seen = frame_id

        self.locked = False
        self.history = [(cx,cy)]

    def apply_h(self, H):
        if H is None:
            return
        pts = np.array([[self.bbox[0], self.bbox[1]],
                        [self.bbox[2], self.bbox[3]]], dtype=np.float32).reshape(-1,1,2)
        pts = cv2.perspectiveTransform(pts, H).reshape(-1,2)
        self.bbox = np.array([pts[0][0], pts[0][1], pts[1][0], pts[1][1]])

    def predict(self, H):
        self.time_since_update += 1

        self.apply_h(H)

        self.bbox[[0,2]] += 0.7*self.velocity[0]
        self.bbox[[1,3]] += 0.7*self.velocity[1]

        self.conf *= 0.96

    def update(self, bbox, feat, frame_id):
        prev_center = self.center.copy()

        self.bbox = bbox[:4].copy()
        self.conf = bbox[4]

        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2

        self.center = np.array([cx,cy])
        self.velocity = 0.6*self.velocity + 0.4*(self.center - prev_center)

        self.history.append((cx,cy))
        if len(self.history) > 40:
            self.history.pop(0)

        if feat is not None:
            self.features.append(feat)
            if len(self.features) > 5:
                self.features.pop(0)
            self.feature = np.mean(self.features, axis=0)

        self.hits += 1
        self.time_since_update = 0
        self.last_seen = frame_id

        if self.hits >= 5:
            self.locked = True


# ================= TRACKER ================= #
class HybridTracker:
    def __init__(self,
                 high_thresh=0.4,
                 match_thresh=0.5,
                 max_lost=20,
                 min_hits=2,
                 window=15):

        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.window = window

        self.tracks = []
        self.memory = []
        self.prev_frame = None
        self.frame_id = 0


    def compute_cost(self, tracks, dets, feats):
        iou = iou_batch(np.array([t.bbox for t in tracks]), dets[:,:4])
        cost = np.zeros_like(iou)

        for i,t in enumerate(tracks):
            for j,f in enumerate(feats):

                sim = cosine_sim(t.feature, f)

                cx = (dets[j,0]+dets[j,2])/2
                cy = (dets[j,1]+dets[j,3])/2
                dist = np.linalg.norm(t.center - np.array([cx,cy]))

                if dist > 120:
                    continue

                motion = np.exp(-dist/100)

                c = 0.5*iou[i,j] + 0.3*sim + 0.2*motion

                if t.locked:
                    c += 0.25   # 🔥 ID protection

                cost[i,j] = c

        return cost


    def recover(self, bbox, feat):
        best_id = None
        best_score = 0

        for m in self.memory:
            dt = self.frame_id - m["last_seen"]
            if dt > self.window:
                continue

            pred = m["bbox"].copy()
            pred[[0,2]] += m["velocity"][0]*dt
            pred[[1,3]] += m["velocity"][1]*dt

            iou = iou_batch(np.array([pred]), np.array([bbox]))[0,0]
            sim = cosine_sim(m["feature"], feat)

            score = 0.5*iou + 0.5*sim

            if score > 0.4 and score > best_score:
                best_score = score
                best_id = m["id"]

        return best_id


    def update(self, detections, features, frame):
        self.frame_id += 1

        if features is None:
            features = [None]*len(detections)

        high = detections[detections[:,4] >= self.high_thresh]
        high_feats = [features[i] for i in range(len(detections))
                      if detections[i,4] >= self.high_thresh]

        H = estimate_homography(self.prev_frame, frame) if self.prev_frame is not None else None

        # predict
        for t in self.tracks:
            t.predict(H)

        # match
        if len(self.tracks) and len(high):
            cost = self.compute_cost(self.tracks, high, high_feats)
            matches, u_tracks, u_dets = hungarian(cost, self.match_thresh)
        else:
            matches, u_tracks, u_dets = [], list(range(len(self.tracks))), list(range(len(high)))

        # update matched
        for ti,di in matches:
            self.tracks[ti].update(high[di], high_feats[di], self.frame_id)

        # new / recover
        for i in u_dets:
            feat = high_feats[i] if i < len(high_feats) else None
            rid = self.recover(high[i][:4], feat)

            if rid is not None:
                t = Track(high[i], feat, self.frame_id)
                t.id = rid
                self.tracks.append(t)
            else:
                if high[i][4] > 0.6:
                    self.tracks.append(Track(high[i], feat, self.frame_id))

        # clean + memory
        alive = []
        for t in self.tracks:
            if t.time_since_update > 2:
                self.memory.append({
                    "id": t.id,
                    "bbox": t.bbox.copy(),
                    "feature": t.feature,
                    "velocity": t.velocity.copy(),
                    "last_seen": t.last_seen
                })

            if t.time_since_update <= self.max_lost and t.conf > 0.25:
                alive.append(t)

        self.tracks = alive

        self.memory = [m for m in self.memory
                       if self.frame_id - m["last_seen"] <= self.window]

        self.prev_frame = frame.copy()

        outputs = []
        for t in self.tracks:
            if t.hits >= self.min_hits and t.time_since_update <= 2:
                outputs.append({
                    "id": t.id,
                    "bbox": t.bbox,
                    "conf": t.conf,
                    "history": t.history
                })

        return outputs


ByteTrackPlus = HybridTracker