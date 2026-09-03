"""
An independent reference implementation of ml/FrameNormalizer.kt.

Deliberately transcribed from the KOTLIN, not from the TypeScript port it is
used to check. If it were derived from the TypeScript it would faithfully
reproduce any mistake in it and the parity test would prove nothing.

Why this test exists at all: FrameNormalizer produces the exact 30x780 buffer
the model consumes. An off-by-one in the index arithmetic does not crash and
does not look wrong on screen — it yields confident, incorrect glosses. Nothing
else in the pipeline would catch it.

Kept structurally identical to the Kotlin (same loop bounds, same order, same
guards) so the two can be diffed by eye as well as by number.
"""

import math

RAW_FEATURES = 258
FINAL_FEATURES = 780
SEQUENCE_LENGTH = 30


def normalize_single_frame(frame: list[float]) -> list[float]:
    """Stage 1 (anchor subtraction) + Stage 2 (scale by shoulder width)."""
    assert len(frame) == RAW_FEATURES, f"Expected {RAW_FEATURES} features, got {len(frame)}"
    f = list(frame)

    # -- Stage 1: Anchor Subtraction --

    # Shoulder landmarks: left=[44..46](xyz), right=[48..50](xyz)
    shoulder_lx, shoulder_ly, shoulder_lz = f[44], f[45], f[46]
    shoulder_rx, shoulder_ry, shoulder_rz = f[48], f[49], f[50]

    # Pose anchor = shoulder midpoint
    anchor_x = (shoulder_lx + shoulder_rx) / 2
    anchor_y = (shoulder_ly + shoulder_ry) / 2
    anchor_z = (shoulder_lz + shoulder_rz) / 2

    # Subtract anchor from pose xyz (33 landmarks x 4, only xyz not visibility)
    if anchor_x != 0 or anchor_y != 0 or anchor_z != 0:
        for i in range(33):
            base = i * 4
            f[base] -= anchor_x
            f[base + 1] -= anchor_y
            f[base + 2] -= anchor_z
            # f[base + 3] = visibility -- unchanged

    # Left hand: anchor to left wrist
    lh_x, lh_y, lh_z = f[132], f[133], f[134]
    if lh_x != 0 or lh_y != 0 or lh_z != 0:
        for i in range(21):
            base = 132 + i * 3
            f[base] -= lh_x
            f[base + 1] -= lh_y
            f[base + 2] -= lh_z

    # Right hand: anchor to right wrist
    rh_x, rh_y, rh_z = f[195], f[196], f[197]
    if rh_x != 0 or rh_y != 0 or rh_z != 0:
        for i in range(21):
            base = 195 + i * 3
            f[base] -= rh_x
            f[base + 1] -= rh_y
            f[base + 2] -= rh_z

    # -- Stage 2: Scale by Shoulder Width --
    # Note the Kotlin reads the ORIGINAL shoulder coordinates here, captured
    # before Stage 1 shifted them. Using the post-anchor values instead would be
    # a silent behaviour change.
    dx = shoulder_lx - shoulder_rx
    dy = shoulder_ly - shoulder_ry
    dz = shoulder_lz - shoulder_rz
    shoulder_width = math.sqrt(dx * dx + dy * dy + dz * dz)

    if shoulder_width > 0.01:
        inv_width = 1 / shoulder_width

        # Scale pose xyz (skip visibility at every 4th element)
        for i in range(33):
            base = i * 4
            f[base] *= inv_width
            f[base + 1] *= inv_width
            f[base + 2] *= inv_width

        # Scale left hand
        for i in range(132, 195):
            f[i] *= inv_width

        # Scale right hand
        for i in range(195, 258):
            f[i] *= inv_width

    return f


def build_sequence_features(sequence: list[list[float]]) -> list[float]:
    """Stages 3 (velocity) + 4 (acceleration) + 5 (engineered), over 30 frames."""
    assert len(sequence) == SEQUENCE_LENGTH, f"Expected {SEQUENCE_LENGTH} frames, got {len(sequence)}"
    t = len(sequence)
    n = RAW_FEATURES

    # Stage 3: Velocity. Frame 0 stays zero -- no earlier frame to difference.
    velocity = [[0.0] * n for _ in range(t)]
    for i in range(1, t):
        for j in range(n):
            velocity[i][j] = sequence[i][j] - sequence[i - 1][j]

    # Stage 4: Acceleration
    acceleration = [[0.0] * n for _ in range(t)]
    for i in range(1, t):
        for j in range(n):
            acceleration[i][j] = velocity[i][j] - velocity[i - 1][j]

    # Stage 5: Engineered features (6 per frame)
    #   left wrist xyz  = [60, 61, 62]  (pose landmark 15)
    #   right wrist xyz = [64, 65, 66]  (pose landmark 16)
    #   nose xyz        = [0, 1, 2]     (pose landmark 0)
    engineered = [[0.0] * 6 for _ in range(t)]
    for i in range(t):
        s = sequence[i]
        lw_x, lw_y, lw_z = s[60], s[61], s[62]
        rw_x, rw_y, rw_z = s[64], s[65], s[66]
        nose_x, nose_y, nose_z = s[0], s[1], s[2]

        # [0] distance between wrists
        dw_x, dw_y, dw_z = lw_x - rw_x, lw_y - rw_y, lw_z - rw_z
        engineered[i][0] = math.sqrt(dw_x * dw_x + dw_y * dw_y + dw_z * dw_z)

        # [1:4] wrist difference vector
        engineered[i][1] = dw_x
        engineered[i][2] = dw_y
        engineered[i][3] = dw_z

        # [4] left wrist to nose
        ln_x, ln_y, ln_z = lw_x - nose_x, lw_y - nose_y, lw_z - nose_z
        engineered[i][4] = math.sqrt(ln_x * ln_x + ln_y * ln_y + ln_z * ln_z)

        # [5] right wrist to nose
        rn_x, rn_y, rn_z = rw_x - nose_x, rw_y - nose_y, rw_z - nose_z
        engineered[i][5] = math.sqrt(rn_x * rn_x + rn_y * rn_y + rn_z * rn_z)

    # [position(258) | velocity(258) | acceleration(258) | engineered(6)] = 780
    result = [0.0] * (t * FINAL_FEATURES)
    for i in range(t):
        offset = i * FINAL_FEATURES
        result[offset : offset + n] = sequence[i]
        result[offset + n : offset + n * 2] = velocity[i]
        result[offset + n * 2 : offset + n * 3] = acceleration[i]
        result[offset + n * 3 : offset + n * 3 + 6] = engineered[i]

    return result
