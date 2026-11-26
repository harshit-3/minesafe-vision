from minesafe.rules import (
    Detection,
    SiteConfig,
    ViolationDebouncer,
    evaluate_frame,
    match_persons,
    point_in_polygon,
)
from minesafe.tracker import IoUTracker

PERSON = Detection("person", (100, 100, 200, 400), 0.9)


def test_compliant_worker():
    dets = [PERSON, Detection("helmet", (130, 100, 170, 140), 0.8), Detection("vest", (110, 180, 190, 280), 0.8)]
    (w,) = evaluate_frame(dets)
    assert w.ppe == {"helmet": "ok", "vest": "ok"}
    assert w.violations == []


def test_missing_helmet_flagged():
    dets = [PERSON, Detection("no_helmet", (130, 100, 170, 140), 0.7), Detection("vest", (110, 180, 190, 280), 0.8)]
    (w,) = evaluate_frame(dets)
    assert w.violations == ["no_helmet"]


def test_helmet_at_feet_does_not_count():
    """A helmet lying on the ground near a worker's boots is not being worn."""
    dets = [PERSON, Detection("helmet", (130, 360, 170, 395), 0.9)]
    (w,) = evaluate_frame(dets)
    assert w.ppe["helmet"] == "unknown"


def test_unknown_is_not_a_violation():
    (w,) = evaluate_frame([PERSON])
    assert w.ppe == {"helmet": "unknown", "vest": "unknown"}
    assert w.violations == []


def test_ppe_assigned_to_correct_worker_in_crowd():
    a = Detection("person", (100, 100, 200, 400), 0.9)
    b = Detection("person", (300, 100, 400, 400), 0.9)
    dets = [a, b, Detection("helmet", (130, 100, 170, 140)), Detection("no_helmet", (330, 100, 370, 140))]
    wa, wb = evaluate_frame(dets, SiteConfig(required_ppe=("helmet",)))
    assert wa.violations == [] and wb.violations == ["no_helmet"]


def test_vehicle_proximity():
    forklift = Detection("forklift", (210, 200, 400, 400), 0.9)  # expanded zone reaches x=115
    (w,) = evaluate_frame([PERSON, forklift], SiteConfig(required_ppe=()))
    assert "vehicle_proximity" in w.violations
    far = Detection("forklift", (900, 200, 1000, 400), 0.9)
    (w,) = evaluate_frame([PERSON, far], SiteConfig(required_ppe=()))
    assert w.violations == []


def test_restricted_zone_uses_feet():
    zone = {"conveyor": [(0, 350), (300, 350), (300, 500), (0, 500)]}
    (w,) = evaluate_frame([PERSON], SiteConfig(required_ppe=(), restricted_zones=zone))
    assert w.violations == ["restricted_zone:conveyor"]


def test_low_confidence_ignored():
    dets = [PERSON, Detection("no_helmet", (130, 100, 170, 140), 0.1)]
    (w,) = evaluate_frame(dets, SiteConfig(required_ppe=("helmet",)))
    assert w.violations == []


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and not point_in_polygon((15, 5), sq)


def test_debouncer_ignores_flicker_and_respects_cooldown():
    deb = ViolationDebouncer(window=10, min_hits=6, cooldown=100)
    bad = Detection("person", (0, 0, 10, 30), track_id=1)
    from minesafe.rules import WorkerStatus
    flicker = [deb.update(i, [WorkerStatus(bad, violations=["no_helmet"])] if i % 3 == 0 else [])
               for i in range(30)]
    assert not any(flicker)
    deb = ViolationDebouncer(window=10, min_hits=6, cooldown=100)
    fired = [e for i in range(150) for e in deb.update(i, [WorkerStatus(bad, violations=["no_helmet"])])]
    assert [e.frame for e in fired] == [5, 105]


def test_tracker_keeps_identity():
    tr = IoUTracker()
    ids = []
    for x in range(0, 50, 5):
        (d,) = tr.update([Detection("person", (x, 0, x + 50, 100))])
        ids.append(d.track_id)
    assert len(set(ids)) == 1


def test_match_persons():
    gt = [Detection("person", (0, 0, 10, 10)), Detection("person", (100, 100, 110, 110))]
    pr = [Detection("person", (101, 101, 111, 111)), Detection("person", (500, 500, 510, 510))]
    assert match_persons(gt, pr) == [(1, 0)]


def test_strict_mode_flags_unconfirmed_ppe():
    (w,) = evaluate_frame([PERSON], SiteConfig(unknown_is_violation=True))
    assert sorted(w.violations) == ["no_helmet", "no_vest"]
    assert w.ppe == {"helmet": "unknown", "vest": "unknown"}


def test_forklift_operator_not_flagged_but_pedestrian_is():
    forklift = Detection("forklift", (300, 150, 500, 420), 0.9)
    driver = Detection("person", (360, 160, 450, 380), 0.9)  # sitting inside the vehicle
    pedestrian = Detection("person", (230, 150, 290, 400), 0.9)  # walking next to it
    wd, wp = evaluate_frame([driver, pedestrian, forklift], SiteConfig(required_ppe=()))
    assert wd.is_operator and wd.violations == []
    assert wp.violations == ["vehicle_proximity"]


def test_live_processor_tracks_and_debounces():
    """LiveProcessor keeps one track per worker and raises one alert for a persistent violation."""
    import numpy as np

    from minesafe.pipeline import LiveProcessor

    class FakeDet:
        conf = 0.35

        def __call__(self, frame):
            return [Detection("person", (100, 100, 200, 400), 0.9),
                    Detection("no_helmet", (130, 100, 170, 140), 0.8),
                    Detection("vest", (110, 180, 190, 280), 0.8)]

    proc = LiveProcessor(FakeDet(), None, SiteConfig(), fps=10)
    frame = np.zeros((480, 640, 3), np.uint8)
    for _ in range(40):
        out, workers, _ = proc.process(frame)
    assert out.shape == frame.shape
    assert workers[0].person.track_id == 1
    assert [e["violation"] for e in proc.events] == ["no_helmet"]


def test_tracker_keeps_identity_for_small_fast_moving_worker():
    """Tiny distant worker (12x30 px) moving 8 px/frame: boxes barely overlap frame to frame."""
    tr = IoUTracker()
    ids = [tr.update([Detection("person", (x, 50, x + 12, 80))])[0].track_id for x in range(0, 200, 8)]
    assert len(set(ids)) == 1


def test_tracker_survives_short_occlusion_and_separates_workers():
    tr = IoUTracker()
    a = lambda x: Detection("person", (x, 0, x + 40, 100))  # noqa: E731
    b = lambda x: Detection("person", (x + 300, 0, x + 340, 100))  # noqa: E731
    first = tr.update([a(0), b(0)])
    for _ in range(5):
        tr.update([b(0)])  # worker A occluded for 5 frames
    again = tr.update([a(10), b(0)])
    assert again[0].track_id == first[0].track_id and again[1].track_id == first[1].track_id
    assert first[0].track_id != first[1].track_id
