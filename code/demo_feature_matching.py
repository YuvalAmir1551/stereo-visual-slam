"""Consensus Matching — Feature detection, matching, and significance testing."""

import random
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from dataset import read_images
from features import extract_features, match_descriptors_knn, apply_ratio_test

DETECTOR = 'ORB'
N_FEATURES = 5000
RATIO_THRESHOLD = 0.7


def plot_keypoints(img_left, img_right, kp_left, kp_right):
    """Display detected keypoints on both stereo images."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.18})

    img_left_kp = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0))
    img_right_kp = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0))

    for ax, img, side, n in [(ax1, img_left_kp, 'Left', len(kp_left)),
                             (ax2, img_right_kp, 'Right', len(kp_right))]:
        ax.imshow(img)
        ax.set_title(f'{side} image — {n} keypoints')
        ax.set_xlabel('x (pixels)')
        ax.set_aspect('equal')
    ax1.set_ylabel('y (pixels)')


def plot_matches(img_left, img_right, kp_left, kp_right, matches,
                 n_display=20, title='Matches'):
    """Display n random matches as lines connecting two separate subplots."""
    sample = random.sample(matches, min(n_display, len(matches)))
    cmap = plt.get_cmap('tab20')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'wspace': 0.18})
    for ax, img, side in [(ax1, img_left, 'Left'), (ax2, img_right, 'Right')]:
        ax.imshow(img, cmap='gray')
        ax.set_title(f'{side} image')
        ax.set_xlabel('x (pixels)')
        ax.set_aspect('equal')
    ax1.set_ylabel('y (pixels)')

    for i, m in enumerate(sample):
        pt_l = kp_left[m.queryIdx].pt
        pt_r = kp_right[m.trainIdx].pt
        color = cmap(i % 20)
        ax1.plot(*pt_l, 'o', color=color, markersize=4)
        ax2.plot(*pt_r, 'o', color=color, markersize=4)
        fig.add_artist(ConnectionPatch(
            xyA=pt_l, coordsA=ax1.transData,
            xyB=pt_r, coordsB=ax2.transData,
            color=color, linewidth=0.8, alpha=0.8))

    fig.suptitle(f'{title} ({len(sample)} shown out of {len(matches)})')


def plot_rejected_match(img_left, img_right, kp_left, kp_right, match, zoom=60):
    """Display a single rejected match as dots on both images, zoomed in."""
    pt_left = kp_left[match.queryIdx].pt
    pt_right = kp_right[match.trainIdx].pt

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'wspace': 0.18})

    for ax, img, pt, side in [(ax1, img_left, pt_left, 'Left'),
                              (ax2, img_right, pt_right, 'Right')]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        ax.plot(pt[0], pt[1], marker='+', color='red', markersize=12,
                markeredgewidth=1.5, linestyle='None')
        ax.set_xlim(pt[0] - zoom, pt[0] + zoom)
        ax.set_ylim(pt[1] + zoom, pt[1] - zoom)  # inverted for image coords
        ax.set_aspect('equal')
        ax.set_title(f'{side} image — rejected match @ ({pt[0]:.1f}, {pt[1]:.1f})')
        ax.set_xlabel('x (pixels)')
    ax1.set_ylabel('y (pixels)')


def q1(img_left, img_right):
    """Q1.1: Detect keypoints and display them on both images."""
    kp_left, desc_left = extract_features(img_left, detector=DETECTOR, n_features=N_FEATURES)
    kp_right, desc_right = extract_features(img_right, detector=DETECTOR, n_features=N_FEATURES)

    print(f"Keypoints detected — Left: {len(kp_left)}, Right: {len(kp_right)}")
    plot_keypoints(img_left, img_right, kp_left, kp_right)

    return kp_left, desc_left, kp_right, desc_right


def q2(desc_left):
    """Q1.2: Print the descriptors of the first two features."""
    print("First descriptor (left image):")
    print(desc_left[0])
    print("\nSecond descriptor (left image):")
    print(desc_left[1])


def q3(img_left, img_right, kp_left, desc_left, kp_right, desc_right):
    """Q1.3: Match descriptors and present 20 random matches."""
    knn_matches = match_descriptors_knn(desc_left, desc_right, detector=DETECTOR, k=2)
    best_matches = [m for m, _ in knn_matches]
    print(f"Total matches: {len(best_matches)}")
    plot_matches(img_left, img_right, kp_left, kp_right, best_matches,
                 n_display=20, title='Random matches (no filtering)')


def q4(img_left, img_right, kp_left, desc_left, kp_right, desc_right):
    """Q1.4: Apply ratio test, show filtered matches, find a correct rejected match."""
    knn_matches = match_descriptors_knn(desc_left, desc_right, detector=DETECTOR, k=2)
    accepted, rejected = apply_ratio_test(knn_matches, ratio=RATIO_THRESHOLD)

    print(f"Ratio threshold: {RATIO_THRESHOLD}")
    print(f"Accepted matches: {len(accepted)}")
    print(f"Rejected (discarded) matches: {len(rejected)}")

    plot_matches(img_left, img_right, kp_left, kp_right, accepted,
                 n_display=20, title=f'Matches after ratio test (ratio={RATIO_THRESHOLD})')

    # A rejected match is considered "correct" if it satisfies the epipolar
    # constraint: in a rectified stereo pair, matching points lie on the same
    # horizontal line (|Δy| ≈ 0). `rejected` is sorted by ratio ascending, so
    # iterating it yields the matches that barely failed the test first —
    # those are the most likely to be truly correct.
    y_threshold = 2.0
    correct_rejected = None
    correct_ratio = None
    for m, r in rejected:
        pt_left = kp_left[m.queryIdx].pt
        pt_right = kp_right[m.trainIdx].pt
        if abs(pt_left[1] - pt_right[1]) < y_threshold:
            correct_rejected = m
            correct_ratio = r
            break

    if correct_rejected is not None:
        pt_l = kp_left[correct_rejected.queryIdx].pt
        pt_r = kp_right[correct_rejected.trainIdx].pt
        print(f"\nFound correct rejected match: left {pt_l} -> right {pt_r}")
        print(f"  y-difference: {abs(pt_l[1] - pt_r[1]):.2f} px")
        print(f"  ratio: {correct_ratio:.3f} (threshold: {RATIO_THRESHOLD})")
        plot_rejected_match(img_left, img_right, kp_left, kp_right, correct_rejected)
    else:
        print("\nNo correct rejected match found — try a stricter ratio threshold.")


def main():
    """Run all questions for Exercise 1."""
    img_left, img_right = read_images(0)

    print("=" * 60)
    print("Q1.1 — Keypoint Detection")
    print("=" * 60)
    kp_left, desc_left, kp_right, desc_right = q1(img_left, img_right)

    print("\n" + "=" * 60)
    print("Q1.2 — Feature Descriptors")
    print("=" * 60)
    q2(desc_left)

    print("\n" + "=" * 60)
    print("Q1.3 — Descriptor Matching")
    print("=" * 60)
    q3(img_left, img_right, kp_left, desc_left, kp_right, desc_right)

    print("\n" + "=" * 60)
    print("Q1.4 — Significance Test (Ratio Test)")
    print("=" * 60)
    q4(img_left, img_right, kp_left, desc_left, kp_right, desc_right)

    plt.show()


if __name__ == '__main__':
    main()
