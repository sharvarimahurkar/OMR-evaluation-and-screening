# omr_utils.py
import cv2
import numpy as np
import base64
from typing import Tuple, Dict, List

# subject ranges (1-based inclusive)
SUBJECT_RANGES = {
    "PYTHON": (1, 20),
    "DATA ANALYSIS": (21, 40),
    "MySQL": (41, 60),
    "POWER BI": (61, 80),
    "Adv STATS": (81, 100),
}

def warp_document(image: np.ndarray, target_w=1200, target_h=1600) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    docCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
    if docCnt is None:
        # fallback: return resized original
        return cv2.resize(image, (target_w, target_h))
    pts = docCnt.reshape(4, 2)
    # order points: tl, tr, br, bl
    def order_pts(pts):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(4)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype="float32")
    rect = order_pts(pts)
    dst = np.array([[0,0],[target_w-1,0],[target_w-1,target_h-1],[0,target_h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_w, target_h))
    return warped

def find_bubble_contours(thresh_img: np.ndarray, min_area=800, max_area=6000) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_cnts = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        ar = w/float(h) if h>0 else 0
        if min_area < area < max_area and 0.7 <= ar <= 1.3:
            bubble_cnts.append(c)
    return bubble_cnts

def group_bubbles_into_questions(bubble_cnts: List[np.ndarray], expected_questions=100, choices_per_q=4):
    # compute centers
    boxes = []
    for c in bubble_cnts:
        x,y,w,h = cv2.boundingRect(c)
        cx = x + w/2
        cy = y + h/2
        boxes.append((c, (int(cx), int(cy)), (x,y,w,h)))
    # sort by y then x
    boxes_sorted = sorted(boxes, key=lambda b: (b[1][1], b[1][0]))
    # group into rows by y proximity
    rows = []
    current_row = []
    row_y = None
    y_thresh = 20  # pixels, tune if needed
    for item in boxes_sorted:
        _, (cx, cy), _ = item
        if row_y is None:
            row_y = cy
            current_row = [item]
        elif abs(cy - row_y) <= y_thresh:
            current_row.append(item)
        else:
            rows.append(current_row)
            current_row = [item]
            row_y = cy
    if current_row:
        rows.append(current_row)
    # flatten rows but ensure we have groups of choices_per_q
    # In some forms, rows will contain many bubbles; we need to split into question rows.
    # Create final list of question rows by splitting each row into chunks of size choices_per_q after sorting by x.
    question_rows = []
    for r in rows:
        # sort row by x
        r_sorted = sorted(r, key=lambda it: it[1][0])
        # split into chunks
        for i in range(0, len(r_sorted), choices_per_q):
            chunk = r_sorted[i:i+choices_per_q]
            if len(chunk) == choices_per_q:
                question_rows.append(chunk)
    # If we didn't detect exactly expected_questions, try alternative: sort all by y then chunk
    if len(question_rows) != expected_questions:
        # fallback: sort centers by y then x and chunk sequentially
        all_sorted = sorted(boxes, key=lambda b: (b[1][1], b[1][0]))
        tmp = [all_sorted[i:i+choices_per_q] for i in range(0, len(all_sorted), choices_per_q)]
        question_rows = [t for t in tmp if len(t)==choices_per_q]
    # final sanity: if more than expected, truncate; if fewer, we return as many as detected
    return question_rows[:expected_questions]

def detect_answers_and_score(image_bytes: bytes, answer_key: List[str]) -> Tuple[int, Dict[str,int], bytes, List[Tuple[int,str,str]]]:
    """
    Returns:
      total_score, per_subject_scores, overlay_bytes (jpg), list_of_question_results ([(q_index, selected_opt, correct_opt), ...])
    """
    # decode image
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    warped = warp_document(img, target_w=1200, target_h=1600)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # threshold - invert: bubbles become white on black
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find bubble contours
    bubble_cnts = find_bubble_contours(thresh, min_area=600, max_area=6000)
    question_rows = group_bubbles_into_questions(bubble_cnts, expected_questions=100, choices_per_q=4)

    results = []
    total = 0

    overlay = warped.copy()
    question_number = 1
    for q_row in question_rows:
        # sort by x
        q_row_sorted = sorted(q_row, key=lambda it: it[1][0])
        # for each choice compute filled area
        filled = []
        for (cnt, (cx,cy), (x,y,w,h)) in q_row_sorted:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            # count white pixels inside contour (since thresh inverted)
            total_pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            # ratio: filled_pixels / total_pixels
            ratio = (filled_pixels / float(total_pixels)) if total_pixels>0 else 0
            filled.append(ratio)
        # find selected: max ratio and above threshold
        max_idx = int(np.argmax(filled))
        max_val = filled[max_idx]
        # threshold to decide if bubble is marked
        SELECTED_THRESHOLD = 0.35  # tune per form
        selected_opt = None
        if max_val >= SELECTED_THRESHOLD:
            selected_opt = ["A","B","C","D"][max_idx]
        else:
            selected_opt = None  # blank / undetected

        correct_opt = answer_key[question_number-1] if question_number-1 < len(answer_key) else None
        is_correct = (selected_opt == correct_opt)
        if selected_opt is not None and is_correct:
            total += 1

        results.append((question_number, selected_opt if selected_opt else "", correct_opt if correct_opt else ""))

        # draw overlay markers
        for idx, (cnt, (cx,cy), (x,y,w,h)) in enumerate(q_row_sorted):
            color = (0,0,255)  # red default
            if idx == max_idx and max_val >= SELECTED_THRESHOLD:
                # selected - green if correct else red
                color = (0,255,0) if (["A","B","C","D"][idx] == correct_opt) else (0,0,255)
            # draw rectangle / contour
            cv2.drawContours(overlay, [cnt], -1, color, 2)
        # annotate question number
        cx_row = int(sum([c[1][0] for c in q_row_sorted]) / len(q_row_sorted))
        cy_row = int(min([c[1][1] for c in q_row_sorted]) - 10)
        cv2.putText(overlay, str(question_number), (cx_row-10, max(20, cy_row)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        question_number += 1

    # compute per subject scores
    per_subject = {}
    for subj, (s,e) in SUBJECT_RANGES.items():
        cnt = 0
        for q_idx, sel, corr in results:
            if s <= q_idx <= e:
                if sel != "" and sel == corr:
                    cnt += 1
        per_subject[subj] = cnt

    # encode overlay to bytes
    ok, buff = cv2.imencode(".jpg", overlay)
    overlay_bytes = buff.tobytes() if ok else image_bytes

    return total, per_subject, overlay_bytes, results
