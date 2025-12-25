import cv2
import numpy as np

def feature_match_homography(template_path: str, image_path: str,
                             nfeatures: int = 2000,
                             ratio: float = 0.75,
                             ransac_reproj_thresh: float = 3.0):
    # 1) Load (grayscale)
    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None or img is None:
        raise FileNotFoundError("Failed to read template or image.")

    # 2) Detect & compute (ORB)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(tmpl, None)
    kp2, des2 = orb.detectAndCompute(img,  None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise RuntimeError("Not enough keypoints/descriptors.")

    # 3) Match (ORB -> Hamming). Use knnMatch + ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_n in knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError(f"Not enough good matches after ratio test: {len(good)}")

    # 4) Homography with RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    # 5) Project template corners onto image
    h1, w1 = tmpl.shape[:2]
    corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # 6) Visualization
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw projected quadrilateral
    proj_int = np.int32(proj)
    cv2.polylines(vis, [proj_int], isClosed=True, color=(0, 255, 0), thickness=2)

    # (optional) draw inlier matches (heavy; good for debug)
    dbg = cv2.drawMatches(
        tmpl, kp1, img, kp2, good, None,
        matchesMask=inlier_mask.ravel().tolist() if inlier_mask is not None else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "H": H,
        "good_matches": len(good),
        "inliers": inliers,
        "proj_corners": proj,  # (4,2)
        "vis_image": vis,
        "debug_matches": dbg
    }

if __name__ == "__main__":
    res = feature_match_homography("template.png", "input.png")

    print("good_matches:", res["good_matches"])
    print("inliers:", res["inliers"])
    print("proj_corners:\n", res["proj_corners"])

    cv2.imwrite("match_result.png", res["vis_image"])
    cv2.imwrite("match_debug.png", res["debug_matches"])