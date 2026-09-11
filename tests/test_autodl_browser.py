from __future__ import annotations

import random

import pytest

from autoresearch.services.autodl_browser import (
    AutoDLBrowserSession,
    AutoDLCredentials,
    PuzzleMatch,
    generate_mouse_trajectory,
    next_slider_distance,
    slider_distance_for_gap,
    solve_puzzle_gap,
)


def _synthetic_challenge(target: int = 71) -> tuple[list[list[int]], list[list[int]]]:
    background_width, height, piece_width = 140, 64, 26
    alpha = [[0] * piece_width for _ in range(height)]
    for y in range(16, 48):
        left = 3 if not 25 <= y <= 36 else 0
        right = 20 if not 20 <= y <= 29 else 24
        for x in range(left, right + 1):
            alpha[y][x] = 255

    background = [
        [110 + x // 17 + y // 13 for x in range(background_width)]
        for y in range(height)
    ]
    for y, row in enumerate(alpha):
        for x, value in enumerate(row):
            if value:
                background[y][target + x] = 35
    return background, alpha


def test_puzzle_solver_finds_translated_contour() -> None:
    background, alpha = _synthetic_challenge(target=71)

    match = solve_puzzle_gap(background, alpha, search_start=15)

    assert match.offset == pytest.approx(71, abs=0.75)
    assert match.confidence > 1.04
    assert match.background_width == 140
    assert match.piece_width == 26


@pytest.mark.parametrize(
    "background,alpha,message",
    [
        ([], [[0]], "background_gray is empty"),
        ([[0, 0]], [[255], [255]], "same height"),
        ([[0, 0]], [[0, 0]], "narrower"),
        ([[0, 0], [0, 0]], [[0], [0]], "visible puzzle shape"),
    ],
)
def test_puzzle_solver_rejects_invalid_geometry(
    background: list[list[int]], alpha: list[list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_puzzle_gap(background, alpha)


def test_slider_distance_uses_rendered_piece_offset_as_initial_pointer_target() -> None:
    match = PuzzleMatch(
        offset=173,
        score=10,
        confidence=2,
        background_width=296,
        piece_width=52,
    )

    distance = slider_distance_for_gap(
        match,
        rendered_background_width=300,
        rendered_piece_width=52,
        track_width=300,
        thumb_width=40,
    )

    assert distance == pytest.approx(175.34, abs=0.05)
    assert distance != pytest.approx(match.offset)


def test_slider_distance_is_clamped_to_thumb_travel() -> None:
    match = PuzzleMatch(
        offset=290,
        score=10,
        confidence=2,
        background_width=296,
        piece_width=52,
    )

    distance = slider_distance_for_gap(
        match,
        rendered_background_width=300,
        rendered_piece_width=52,
        track_width=300,
        thumb_width=40,
    )

    assert distance == 260


def test_closed_loop_slider_correction_uses_observed_piece_position() -> None:
    correction = next_slider_distance(
        175.34,
        desired_piece_left=175.34,
        actual_piece_left=151.2,
        maximum_distance=259,
    )

    assert correction == pytest.approx(199.48)
    assert next_slider_distance(
        correction,
        desired_piece_left=175.34,
        actual_piece_left=174.5,
        maximum_distance=259,
    ) is None


def test_mouse_trajectory_is_curved_variable_speed_and_exact() -> None:
    points = generate_mouse_trajectory(
        (20.0, 30.0), (210.0, 30.0), rng=random.Random(7), steps=40
    )

    assert len(points) == 40
    assert points[-1].x == 210.0
    assert points[-1].y == 30.0
    assert max(point.x for point in points) > 210.0
    assert max(abs(point.y - 30.0) for point in points) > 0.5
    distances = [
        abs(current.x - previous.x)
        for previous, current in zip(points, points[1:])
    ]
    assert max(distances) > min(distances) * 4
    assert len({round(point.delay_seconds, 4) for point in points}) > 8


def test_credentials_repr_does_not_expose_runtime_secrets() -> None:
    credentials = AutoDLCredentials(phone="test-phone", password="test-password")

    rendered = repr(credentials)

    assert "test-phone" not in rendered
    assert "test-password" not in rendered


@pytest.mark.asyncio
async def test_navigation_rejects_non_autodl_hosts_before_browser_access() -> None:
    session = AutoDLBrowserSession()

    with pytest.raises(ValueError, match="trusted AutoDL"):
        await session.goto_autodl("https://example.com/collect")
