/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Encoder-independent rate-control policy.

/// Size the CBR VBV/HRD buffer as a multiple of one frame's bit budget.
#[must_use]
pub fn vbv_bits(bitrate_bps: u32, fps: f64, keyframe_interval_s: f64, multiplier: f64) -> u32 {
    let frame_bits = f64::from(bitrate_bps) / fps.max(1.0);
    let selected_multiplier = if multiplier > 0.0 {
        multiplier
    } else if keyframe_interval_s > 0.0 {
        3.0
    } else {
        1.5
    };
    (frame_bits * selected_multiplier)
        .round()
        .max(1.0)
        .min(f64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::vbv_bits;

    #[test]
    fn defaults_match_pixelflux_policy() {
        assert_eq!(vbv_bits(6_000_000, 60.0, 0.0, 0.0), 150_000);
        assert_eq!(vbv_bits(6_000_000, 60.0, 2.0, 0.0), 300_000);
        assert_eq!(vbv_bits(6_000_000, 60.0, 2.0, 2.0), 200_000);
    }

    #[test]
    fn invalid_rates_are_bounded() {
        assert_eq!(vbv_bits(0, 0.0, 0.0, 0.0), 1);
        assert_eq!(vbv_bits(u32::MAX, 1.0, 1.0, f64::MAX), u32::MAX);
    }
}
